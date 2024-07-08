#!/usr/bin/env python3

import rospy
import os
import numpy as np
import torch
import message_filters
import cv_bridge
from pathlib import Path
import open3d as o3d
import tf2_ros
from tf.transformations import quaternion_matrix
import cv2
import time

from utils import *
from adet.config import get_cfg
from adet.utils.visualizer import visualize_pred_amoda_occ
from adet.utils.post_process import detector_postprocess, DefaultPredictor
from foreground_segmentation.model import Context_Guided_Network

from std_msgs.msg import Int32, Header 
from sensor_msgs.msg import Image, CameraInfo, RegionOfInterest, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from uoais.msg import UOAISResults
from uoais.srv import UOAISRequest, UOAISRequestResponse


class UOAIS():

    def __init__(self):

        rospy.init_node("uoais")
        """
        get ros parameters
        """
        self.mode = rospy.get_param("~mode", "topic") 
        rospy.loginfo("Starting uoais node with {} mode".format(self.mode))
        self.rgb_topic = rospy.get_param("~rgb", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth", "/camera/aligned_depth_to_color/image_raw")
        camera_info_topic = rospy.get_param("~camera_info", "/camera/color/camera_info")
        # UOAIS-Net
        self.det2_config_file = rospy.get_param("~config_file", 
                            "configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml")
        self.confidence_threshold = rospy.get_param("~confidence_threshold", 0.5)
        # CG-Net foreground segmentation
        self.use_cgnet = rospy.get_param("~use_cgnet", False)
        self.cgnet_weight = rospy.get_param("~cgnet_weight",
                             "foreground_segmentation/rgbd_fg.pth")
        # RANSAC plane segmentation
        self.use_planeseg = rospy.get_param("~use_planeseg", False)
        self.ransac_threshold = rospy.get_param("~ransac_threshold", 0.003)
        self.ransac_n = rospy.get_param("~ransac_n", 3)
        self.ransac_iter = rospy.get_param("~ransac_iter", 10)
        """
        end of ros parameters
        """

        # initialize cv bridge
        self.cv_bridge = cv_bridge.CvBridge()
        
        # initialize UOAIS-Net and CG-Net
        self.load_models()

        # tf listener
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)  # Create a tf listener

        self.inferencing = False
        self.target_center = None  # x, y and z center of target point cloud
        self.target_pointcloud = None  # point cloud of target
        self.target_amodal_mask = None  # amodal mask of target

        # if use_planeseg, get camera intrinsic
        if self.use_planeseg:
            camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)
            self.K = camera_info.K
            self.o3d_camera_intrinsic = None

        # decide the mode of operation
        self.start_sub = rospy.Subscriber('uoais/start', Int32, self.start_callback, queue_size=1)
        self.start_sub = rospy.Subscriber('uoais/end', Int32, self.end_callback, queue_size=1)
        if self.mode == "topic":
            rgb_sub = message_filters.Subscriber(self.rgb_topic, Image)
            depth_sub = message_filters.Subscriber(self.depth_topic, Image)
            self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=1, slop=2)
            self.ts.registerCallback(self.topic_callback)
            self.result_pub = rospy.Publisher("/uoais/results", UOAISResults, queue_size=10)
            rospy.loginfo("uoais results at topic: /uoais/results")
        elif self.mode == "service":
            self.srv = rospy.Service('/get_uoais_results', UOAISRequest, self.service_callback)
            rospy.loginfo("uoais results at service: /get_uoais_results")
        else:
            raise NotImplementedError
        
        # publishers
        rospy.loginfo("publishing the results of uoais node : /uoais/target_pcd, targetmask_img, vis_img")
        self.vis_pub = rospy.Publisher("/uoais/vis_img", Image, queue_size=10)
        self.mask_img_pub = rospy.Publisher("/uoais/targetmask_img", Image, queue_size=10)
        self.pcd_pub = rospy.Publisher("/uoais/target_pcd", PointCloud2, queue_size=10)
        rospy.loginfo("publishing /uoais/start to start uoais node and /uoais/end to end uoais node")



    def load_models(self):

        # UOAIS-Net
        self.det2_config_file = os.path.join(Path(__file__).parent.parent, self.det2_config_file)
        rospy.loginfo("Loading UOAIS-Net with config_file: {}".format(self.det2_config_file))
        self.cfg = get_cfg()
        self.cfg.merge_from_file(self.det2_config_file)
        self.cfg.defrost()
        self.cfg.MODEL.WEIGHTS = os.path.join(Path(__file__).parent.parent, self.cfg.OUTPUT_DIR, "model_final.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.confidence_threshold
        self.predictor = DefaultPredictor(self.cfg)
        self.W, self.H = self.cfg.INPUT.IMG_SIZE

        # CG-Net (foreground segmentation)
        if self.use_cgnet:
            checkpoint = torch.load(os.path.join(Path(__file__).parent.parent, self.cgnet_weight))
            self.fg_model = Context_Guided_Network(classes=2, in_channel=4)
            self.fg_model.load_state_dict(checkpoint['model'])
            self.fg_model.cuda()
            self.fg_model.eval()


    def topic_callback(self, rgb_msg, depth_msg):
        # self.rgb_msg = rgb_msg
        # self.depth_msg = depth_msg
        if self.inferencing:
            results = self.inference(rgb_msg, depth_msg)        
            self.result_pub.publish(results)

    def service_callback(self, msg):
        if self.inferencing:
            rgb_msg = rospy.wait_for_message(self.rgb_topic, Image)
            depth_msg = rospy.wait_for_message(self.depth_topic, Image)
            results = self.inference(rgb_msg, depth_msg)        
        return UOAISRequestResponse(results)
    
    def start_callback(self, msg):
        self.inferencing = True
        # results = self.inference(self.rgb_msg, self.depth_msg)
        # if self.mode == "topic":
        #     self.result_pub.publish(results)
        # elif self.mode == "service":
        #     return UOAISRequestResponse(results)
    
    def end_callback(self, msg):
        self.inferencing = False
        self.target_pointcloud = None 

    def inference(self, rgb_msg, depth_msg):
        start_time = time.time()
        rgb_img = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        ori_H, ori_W, _ = rgb_img.shape
        rgb_img = cv2.resize(rgb_img, (self.W, self.H))
        depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        depth_img = normalize_depth(depth)
        depth_img = cv2.resize(depth_img, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        depth_img = inpaint_depth(depth_img)

        # UOAIS-Net inference
        if self.cfg.INPUT.DEPTH and self.cfg.INPUT.DEPTH_ONLY:
            uoais_input = depth_img
        elif self.cfg.INPUT.DEPTH and not self.cfg.INPUT.DEPTH_ONLY: 
            uoais_input = np.concatenate([rgb_img, depth_img], -1)        
        outputs = self.predictor(uoais_input)
        instances = detector_postprocess(outputs['instances'], self.H, self.W).to('cpu')
        preds = instances.pred_masks.detach().cpu().numpy()
        pred_visibles = instances.pred_visible_masks.detach().cpu().numpy() 
        pred_bboxes = instances.pred_boxes.tensor.detach().cpu().numpy() 
        pred_occs = instances.pred_occlusions.detach().cpu().numpy() 

        # filter out the background instances 
        # CG-Net
        if self.use_cgnet:
            rospy.loginfo_once("Using foreground segmentation model (CG-Net) to filter out background instances")
            fg_rgb_input = standardize_image(cv2.resize(rgb_img, (320, 240)))
            fg_rgb_input = array_to_tensor(fg_rgb_input).unsqueeze(0)
            fg_depth_input = cv2.resize(depth_img, (320, 240)) 
            fg_depth_input = array_to_tensor(fg_depth_input[:,:,0:1]).unsqueeze(0) / 255
            fg_input = torch.cat([fg_rgb_input, fg_depth_input], 1)
            fg_output = self.fg_model(fg_input.cuda())
            fg_output = fg_output.cpu().data[0].numpy().transpose(1, 2, 0)
            fg_output = np.asarray(np.argmax(fg_output, axis=2), dtype=np.uint8)
            fg_output = cv2.resize(fg_output, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

        # RANSAC
        if self.use_planeseg:
            rospy.loginfo_once("Using RANSAC plane segmentation to filter out background instances")
            o3d_rgb_img = o3d.geometry.Image(rgb_img)
            o3d_depth_img = o3d.geometry.Image(unnormalize_depth(depth_img))
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb_img, o3d_depth_img)
            if self.o3d_camera_intrinsic is None:
                self.o3d_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                                            self.W, self.H, 
                                            self.K[0]*self.W/ori_W, 
                                            self.K[4]*self.H/ori_H, 
                                            self.K[2], self.K[5])
            o3d_pc = o3d.geometry.PointCloud.create_from_rgbd_image(
                                                rgbd_image, self.o3d_camera_intrinsic)
            plane_model, inliers = o3d_pc.segment_plane(distance_threshold=self.ransac_threshold,
                                                        ransac_n=self.ransac_n,
                                                        num_iterations=self.ransac_iter)
            fg_output = np.ones(self.H * self.W)
            fg_output[inliers] = 0 
            fg_output = np.resize(fg_output, (self.H, self.W))
            fg_output = np.uint8(fg_output)

        if self.use_cgnet or self.use_planeseg:
            remove_idxs = []
            for i, pred_visible in enumerate(pred_visibles):
                iou = np.sum(np.bitwise_and(pred_visible, fg_output)) / np.sum(pred_visible)
                if iou < 0.5: 
                    remove_idxs.append(i)
            preds = np.delete(preds, remove_idxs, 0)
            pred_visibles = np.delete(pred_visibles, remove_idxs, 0)
            pred_bboxes = np.delete(pred_bboxes, remove_idxs, 0)
            pred_occs = np.delete(pred_occs, remove_idxs, 0)
        
        # reorder predictions for visualization
        idx_shuf = np.concatenate((np.where(pred_occs==1)[0] , np.where(pred_occs==0)[0] )) 
        preds, pred_visibles, pred_occs, pred_bboxes = \
            preds[idx_shuf], pred_visibles[idx_shuf], pred_occs[idx_shuf], pred_bboxes[idx_shuf]
        vis_img = visualize_pred_amoda_occ(rgb_img, preds, pred_bboxes, pred_occs)
        if self.use_cgnet or self.use_planeseg:
            vis_fg = np.zeros_like(rgb_img) 
            vis_fg[:, :, 1] = fg_output * 255
            vis_img = cv2.addWeighted(vis_img, 0.8, vis_fg, 0.2, 0)
        self.vis_pub.publish(self.cv_bridge.cv2_to_imgmsg(vis_img, encoding="bgr8"))

        # pubish the uoais results
        results = UOAISResults()
        results.header = rgb_msg.header
        results.bboxes = []
        results.visible_masks = []
        results.amodal_masks = []
        n_instances = len(pred_occs)
        for i in range(n_instances):
            bbox = RegionOfInterest()
            bbox.x_offset = int(pred_bboxes[i][0])
            bbox.y_offset = int(pred_bboxes[i][1])
            bbox.width = int(pred_bboxes[i][2]-pred_bboxes[i][0])
            bbox.height = int(pred_bboxes[i][3]-pred_bboxes[i][1])
            results.bboxes.append(bbox)
            results.visible_masks.append(self.cv_bridge.cv2_to_imgmsg(
                                        np.uint8(pred_visibles[i]), encoding="mono8"))
            results.amodal_masks.append(self.cv_bridge.cv2_to_imgmsg(
                                        np.uint8(preds[i]), encoding="mono8"))
        results.occlusions = pred_occs.tolist()
        results.class_names = ["object"] * n_instances
        results.class_ids = [0] * n_instances

        """
        images and point clouds preprocessing
        """
        # get the transform between camera_color_optical_frame and base_link
        try:
            transform_stamped = self.tf_buffer.lookup_transform('camera_link', depth_msg.header.frame_id, rospy.Time(0))
            trans = np.array([transform_stamped.transform.translation.x,
                              transform_stamped.transform.translation.y,
                              transform_stamped.transform.translation.z])
            quat = np.array([transform_stamped.transform.rotation.x,
                            transform_stamped.transform.rotation.y,
                            transform_stamped.transform.rotation.z,
                            transform_stamped.transform.rotation.w])
            T = quaternion_matrix(quat)
            T[:3, 3] = trans
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Failed to transform point cloud: {}".format(e))
            return
        
        """
        use mask to get point cloud and process the raw image
        """
        # create a list to store the point clouds and images after maskeing
        pointcloud_list = []
        dis_list = []
        # camera intrinsic
        self.o3d_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                                        640, 480,
                                        617.0441284179688*640/ori_W,
                                        617.0698852539062*480/ori_H,
                                        322.3338317871094, 238.7687225341797)
        # coord_frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

        # get the point cloud list
        for i in range(n_instances):
            mask = np.array(pred_visibles[i]).astype(int)
            # masking the depth and transform it to point clouds
            un_depth_img = unnormalize_depth(depth_img)
            un_depth_img[np.logical_not(mask)] = 0
            kernel = np.ones((3, 3), np.uint8)
            un_depth_img = cv2.erode(un_depth_img, kernel, iterations=1)
            o3d_rgb_img = o3d.geometry.Image(rgb_img)
            o3d_depth_img = o3d.geometry.Image(un_depth_img)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb_img, o3d_depth_img)
            o3d_pc = o3d.geometry.PointCloud.create_from_rgbd_image(
                                                rgbd_image, self.o3d_camera_intrinsic)
            sampled_pc = np.asarray(o3d_pc.points)
            sampled_o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sampled_pc))
            pcd_filtered, _ = sampled_o3d_pc.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
            pointcloud_list.append(pcd_filtered)

        # decide the target point cloud and image
        # target_pointcloud
        if self.target_pointcloud is None:
            dis_list = [np.linalg.norm(x.get_center()) for x in pointcloud_list]  # get distance between point cloud and origina
            index = np.argmin(dis_list)  # choose the closet one as target
            self.target_pointcloud = pointcloud_list[index].transform(T)  # convert the target point cloud to world frame
            self.target_amodal_mask = np.array(preds[index]).astype(int)  # get the amodal mask of target point cloud
            self.target_center = self.target_pointcloud.get_center()  # get point center of target point cloud in world frame
        else:
            tran_pc_list = [x.transform(T) for x in pointcloud_list]
            dis_list = [np.linalg.norm(x.get_center() - self.target_center) for x in tran_pc_list]
            index = np.argmin(dis_list)
            closet_pointcloud = tran_pc_list[index]
            # self.target_pointcloud.paint_uniform_color([1, 0, 0])
            # closet_pointcloud.paint_uniform_color([0, 1, 0])
            # pointclouds = [self.target_pointcloud, closet_pointcloud]  # compare the preview pc and current one
            # o3d.visualization.draw_geometries(pointclouds)  # visualize the preview pc and current one
            self.target_pointcloud = closet_pointcloud  # update the target point cloud
            self.target_center = self.target_pointcloud.get_center()  # update the center of target point cloud in world frame
        self.target_ros_pointcloud = self.o3d_to_ros(self.target_pointcloud)  # convert the target point cloud to ros format

        # target_image
        masked_img = cv2.resize(rgb_img, (self.W, self.H))
        amodel_mask = np.array(preds[index]).astype(int)
        bounding_box = pred_bboxes[index]
        xi, yi = int(bounding_box[0]), int(bounding_box[1])
        xf, yf = int(bounding_box[2]), int(bounding_box[3])
        # print("bounding box: ", xi, yi, xf, yf)
        masked_img[np.logical_not(amodel_mask)] = 0
        masked_img = masked_img[yi:yf, xi:xf]
        masked_img = cv2.resize(masked_img, (self.W, self.H))

        end_time = time.time()
        print("cost time: {}".format(end_time - start_time))

        self.mask_img_pub.publish(self.cv_bridge.cv2_to_imgmsg(masked_img, encoding="bgr8"))
        self.pcd_pub.publish(self.target_ros_pointcloud)

        return results
    
    def o3d_to_ros(self, point_cloud):
        header = Header()   
        header.stamp = rospy.Time.now()
        header.frame_id = 'camera_link'

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]

        points = np.asarray(point_cloud.points)
        cloud = pc2.create_cloud(header, fields, points)
        return cloud

if __name__ == '__main__':

    uoais = UOAIS()
    rospy.spin() 


#!/usr/bin/env python3

# import rospy
# import os
# import numpy as np
# import torch
# import message_filters
# import cv_bridge
# from pathlib import Path
# import open3d as o3d
# import tf2_ros
# from tf.transformations import quaternion_matrix
# import cv2
# import matplotlib.pyplot as plt
# import time
# import copy
# import itertools
# from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation

# from utils import *
# from adet.config import get_cfg
# from adet.utils.visualizer import visualize_pred_amoda_occ
# from adet.utils.post_process import detector_postprocess, DefaultPredictor
# from foreground_segmentation.model import Context_Guided_Network

# from sensor_msgs.msg import PointCloud2, PointField
# from std_msgs.msg import Int32
# from sensor_msgs.point_cloud2 import create_cloud_xyz32
# from grasp_msg.srv import GraspSegment, GraspSegmentResponse
# from geometry_msgs.msg import TransformStamped
# from std_msgs.msg import String, Header, Float32MultiArray
# from sensor_msgs.msg import Image, CameraInfo, RegionOfInterest
# from uoais.msg import UOAISResults
# from uoais.srv import UOAISRequest, UOAISRequestResponse
# from std_srvs.srv import Empty ,EmptyResponse  



# class UOAIS():

#     def __init__(self):

#         rospy.init_node("uoais")

#         self.mode = rospy.get_param("~mode", "topic")
#         rospy.loginfo("Starting uoais node with {} mode".format(self.mode))
#         self.rgb_topic = rospy.get_param("~rgb", "/camera/color/image_raw")
#         self.depth_topic = rospy.get_param("~depth", "/camera/aligned_depth_to_color/image_raw")
#         # self.depth_topic = rospy.get_param("~depth", "/camera/depth/image_rect_raw")
#         camera_info_topic = rospy.get_param("~camera_info", "/camera/color/camera_info")
#         # UOAIS-Net
#         self.det2_config_file = rospy.get_param(
#             "~config_file",
#             "configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml")
#         self.confidence_threshold = rospy.get_param("~confidence_threshold", 0.5)
#         # CG-Net foreground segmentation
#         self.use_cgnet = rospy.get_param("~use_cgnet", False)
#         self.cgnet_weight = rospy.get_param(
#             "~cgnet_weight",
#             "foreground_segmentation/rgbd_fg.pth")
#         # RANSAC plane segmentation
#         self.use_planeseg = rospy.get_param("~use_planeseg", False)
#         self.ransac_threshold = rospy.get_param("~ransac_threshold", 0.003)
#         self.ransac_n = rospy.get_param("~ransac_n", 3)
#         self.ransac_iter = rospy.get_param("~ransac_iter", 10)
#         self.use_contact = rospy.get_param("~use_contact", False)

#         self.cv_bridge = cv_bridge.CvBridge()
#         self.count = 0
#         self.vis =  False
#         self.choose_mode = "3d" # This is for choosing method, 2d for area, 3d for closest to predefined point
#         # initialize UOAIS-Net and CG-Net
#         self.load_models()
#         """
#         add by matt, do some tf2 work to transform frame
#         """
#         self.tf_buffer = tf2_ros.Buffer()
#         self.listener = tf2_ros.TransformListener(self.tf_buffer)  # Create a tf listener
#         self.target_center = None  # x, y and z center of target point cloud
#         self.target_pointcloud = None  # point cloud of target
#         self.scene_pointcloud = None  # point cloud of target
#         camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)
#         self.K = np.array(camera_info.K)
#         self.o3d_camera_intrinsic = None
#         self.intrinsic = self.K.reshape(3, 3)
#         self.bounds = [[-0.5, 0.3], [-1.5, 1.5], [-0.1, 2]]  # set the bounds
#         bounding_box_points = list(itertools.product(*self.bounds))  # create limit points
#         self.bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
#             o3d.utility.Vector3dVector(bounding_box_points))  # create bounding box object

#         if self.mode == "service":
#             # self.srv = rospy.Service('/get_uoais_results', UOAISRequest, self.service_callback)
#             self.srv = rospy.Service('/get_uoais_results', GraspSegment, self.service_callback)
#             self.points_pub = rospy.Publisher("/uoais/Pointclouds", PointCloud2, queue_size=10)
#             self.obs_points_pub = rospy.Publisher("/uoais/obs_pc", PointCloud2, queue_size=10)
#             self.start_sub = rospy.Subscriber('/uoais/data_init', Int32, self.init_callback, queue_size=1)
#             rospy.loginfo("uoais results at service: /get_uoais_results")
#         else:
#             raise NotImplementedError
#         self.vis_pub = rospy.Publisher("/uoais/vis_img", Image, queue_size=10)
#         self.shutdown_service = rospy.Service('shutdown_uoais', Empty, self.handle_shutdown)
#         rospy.loginfo("visualization results at topic: /uoais/vis_img")

#     def handle_shutdown(self, req):
#         rospy.loginfo("Shutdown service called, shutting down...")
#         rospy.signal_shutdown("Shutdown requested")
#         return EmptyResponse()

#     def load_models(self):
#         # UOAIS-Net
#         self.det2_config_file = os.path.join(Path(__file__).parent.parent, self.det2_config_file)
#         rospy.loginfo("Loading UOAIS-Net with config_file: {}".format(self.det2_config_file))
#         self.cfg = get_cfg()
#         self.cfg.merge_from_file(self.det2_config_file)
#         self.cfg.defrost()
#         self.cfg.MODEL.WEIGHTS = os.path.join(Path(__file__).parent.parent, self.cfg.OUTPUT_DIR, "model_final.pth")
#         self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
#         self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.confidence_threshold
#         self.predictor = DefaultPredictor(self.cfg)
#         self.W, self.H = self.cfg.INPUT.IMG_SIZE

#         # CG-Net (foreground segmentation)
#         if self.use_cgnet:
#             checkpoint = torch.load(os.path.join(Path(__file__).parent.parent, self.cgnet_weight))
#             self.fg_model = Context_Guided_Network(classes=2, in_channel=4)
#             self.fg_model.load_state_dict(checkpoint['model'])
#             self.fg_model.cuda()
#             self.fg_model.eval()

#     def service_callback(self, msg):
#         # rgb_msg = rospy.wait_for_message(self.rgb_topic, Image)
#         # depth_msg = rospy.wait_for_message(self.depth_topic, Image)
#         # target, scene = self.inference(rgb_msg, depth_msg)
#         # return GraspSegmentResponse(target, scene)
#         rospy.loginfo("Clear point cloud data!!!")
#         self.target_center = None  # x, y and z center of target point cloud
#         self.target_pointcloud = None  # point cloud of target
#         self.scene_pointcloud = None  # point cloud of target
#         self.count = 0
#         return GraspSegmentResponse()


#     def init_callback(self, msg):
#         print("init: ",msg.data)
#         if msg.data == 0:
#             rospy.loginfo("Clear point cloud data!!!")
#             self.target_center = None  # x, y and z center of target point cloud
#             self.target_pointcloud = None  # point cloud of target
#             self.scene_pointcloud = None  # point cloud of target
#             self.count = 0
#         elif msg.data == 1:
#             rgb_msg = rospy.wait_for_message(self.rgb_topic, Image)
#             depth_msg = rospy.wait_for_message(self.depth_topic, Image)
#             _ = self.inference(rgb_msg, depth_msg)
#         elif msg.data == 2:
#             rgb_msg = rospy.wait_for_message(self.rgb_topic, Image)
#             depth_msg = rospy.wait_for_message(self.depth_topic, Image)
#             _ = self.inference(rgb_msg, depth_msg, frame="camera")

#     def inference(self, rgb_msg, depth_msg, frame="base"):
#         start_time = time.time()
#         try:
#             """
#             convert camera_color_optical_frame to camer_link
#             """
#             transform_stamped = self.tf_buffer.lookup_transform('base', depth_msg.header.frame_id, rospy.Time(0))
#             trans = np.array([transform_stamped.transform.translation.x,
#                               transform_stamped.transform.translation.y,
#                               transform_stamped.transform.translation.z])
#             quat = np.array([transform_stamped.transform.rotation.x,
#                             transform_stamped.transform.rotation.y,
#                             transform_stamped.transform.rotation.z,
#                             transform_stamped.transform.rotation.w])
#             T = quaternion_matrix(quat)
#             T[:3, 3] = trans
#             T_inv = np.linalg.inv(T)
#         except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
#             rospy.logerr("Failed to transform point cloud: {}".format(e))
#             return

#         self.count += 1
#         rgb_img = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
#         ori_H, ori_W, _ = rgb_img.shape
#         rgb_img = cv2.resize(rgb_img, (self.W, self.H))

#         depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
#         depth_img = normalize_depth(depth)
#         depth_img = cv2.resize(depth_img, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
#         depth_img = inpaint_depth(depth_img)

#         print("0 time: {}".format(time.time() - start_time))

#         # UOAIS-Net inference
#         if self.cfg.INPUT.DEPTH and self.cfg.INPUT.DEPTH_ONLY:
#             uoais_input = depth_img
#         elif self.cfg.INPUT.DEPTH and not self.cfg.INPUT.DEPTH_ONLY:
#             uoais_input = np.concatenate([rgb_img, depth_img], -1)
#         outputs = self.predictor(uoais_input)
#         instances = detector_postprocess(outputs['instances'], self.H, self.W).to('cuda')
#         preds = instances.pred_masks.detach().cpu().numpy()
#         pred_visibles = instances.pred_visible_masks.detach().cpu().numpy()
#         pred_bboxes = instances.pred_boxes.tensor.detach().cpu().numpy()
#         pred_occs = instances.pred_occlusions.detach().cpu().numpy()
#         print("1 time: {}".format(time.time() - start_time))

#         # filter out the background instances
#         # CG-Net
#         o3d_rgb_img = o3d.geometry.Image(rgb_img)
#         o3d_depth_img = o3d.geometry.Image(unnormalize_depth(depth_img))
#         rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb_img, o3d_depth_img)
#         if self.o3d_camera_intrinsic is None:
#             self.o3d_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
#                                         self.W, self.H,
#                                         self.K[0] * self.W / ori_W,
#                                         self.K[4] * self.H / ori_H,
#                                         self.K[2], self.K[5])
#         o3d_pc = o3d.geometry.PointCloud.create_from_rgbd_image(
#             rgbd_image, self.o3d_camera_intrinsic)
#         unorder_pc = np.asarray(o3d_pc.points)

#         if self.use_cgnet:
#             rospy.loginfo_once("Using foreground segmentation model (CG-Net) to filter out background instances")
#             fg_rgb_input = standardize_image(cv2.resize(rgb_img, (320, 240)))
#             fg_rgb_input = array_to_tensor(fg_rgb_input).unsqueeze(0)
#             fg_depth_input = cv2.resize(depth_img, (320, 240))
#             fg_depth_input = array_to_tensor(fg_depth_input[:, :, 0:1]).unsqueeze(0) / 255
#             fg_input = torch.cat([fg_rgb_input, fg_depth_input], 1)
#             fg_output = self.fg_model(fg_input.cuda())
#             fg_output = fg_output.cpu().data[0].numpy().transpose(1, 2, 0)
#             fg_output = np.asarray(np.argmax(fg_output, axis=2), dtype=np.uint8)
#             fg_output = cv2.resize(fg_output, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

#         # RANSAC
#         if self.use_planeseg:
#             rospy.loginfo_once("Using RANSAC plane segmentation to filter out background instances")
#             plane_model, inliers = o3d_pc.segment_plane(distance_threshold=self.ransac_threshold,
#                                                         ransac_n=self.ransac_n,
#                                                         num_iterations=self.ransac_iter)
#             fg_output = np.ones(self.H * self.W)
#             fg_output[inliers] = 0
#             fg_output = np.resize(fg_output, (self.H, self.W))
#             fg_output = np.uint8(fg_output)

#         if self.use_cgnet or self.use_planeseg:
#             remove_idxs = []
#             for i, pred_visible in enumerate(pred_visibles):
#                 iou = np.sum(np.bitwise_and(pred_visible, fg_output)) / np.sum(pred_visible)
#                 if iou < 0.5:
#                     remove_idxs.append(i)
#             preds = np.delete(preds, remove_idxs, 0)
#             pred_visibles = np.delete(pred_visibles, remove_idxs, 0)
#             pred_bboxes = np.delete(pred_bboxes, remove_idxs, 0)
#             pred_occs = np.delete(pred_occs, remove_idxs, 0)

#         # reorder predictions for visualization
#         idx_shuf = np.concatenate((np.where(pred_occs == 1)[0], np.where(pred_occs == 0)[0]))
#         preds, pred_visibles, pred_occs, pred_bboxes = \
#             preds[idx_shuf], pred_visibles[idx_shuf], pred_occs[idx_shuf], pred_bboxes[idx_shuf]
#         vis_img = visualize_pred_amoda_occ(rgb_img, preds, pred_bboxes, pred_occs)

#         if self.use_cgnet or self.use_planeseg:
#             vis_fg = np.zeros_like(rgb_img)
#             vis_fg[:, :, 1] = fg_output * 255
#             vis_img = cv2.addWeighted(vis_img, 0.8, vis_fg, 0.2, 0)

#         self.vis_pub.publish(self.cv_bridge.cv2_to_imgmsg(vis_img, encoding="bgr8"))
#         # pubish the uoais results
#         print("2 time: {}".format(time.time() - start_time))

#         un_depth_img = unnormalize_depth(depth_img)[..., 0] / 1000
#         depth_mask = (un_depth_img > 0.03)
#         final_mask_target = None

#         def backproject_pc(target_mask, frame="base"):
#             target_pc = o3d.geometry.PointCloud()
#             target_pc.points = o3d.utility.Vector3dVector(unorder_pc[target_mask.flatten()])
#             if frame == "base":
#                 target_pc.transform(T)
#                 target_pc = target_pc.crop(self.bounding_box)
#             return target_pc

#         def ratio_pc_concat(mask_1, mask_2, mix_ratio=1, frame="base"):
#             pc_1 = backproject_pc(mask_1, frame=frame)
#             pc_2 = backproject_pc(mask_2, frame=frame)
#             new_pc_2 = regularize_pc_point_count(
#                 np.asarray(pc_2.points), int(len(pc_1.points) * mix_ratio), use_farthest_point=False)

#             pc_1.points = o3d.utility.Vector3dVector(
#                 np.concatenate([new_pc_2, np.asarray(pc_1.points)], axis=0))
#             return pc_1

#         def accu_point_cloud(new_points, acc_points, target_size=10240):
#             downsample_new = regularize_pc_point_count(
#                 np.asarray(new_points.points),
#                 # int(target_size * (0.95 ** self.count)),
#                 int(len(new_points.points) * (0.95 ** self.count)),
#                 use_farthest_point=False)
#             merged_points = np.concatenate([downsample_new, np.asarray(acc_points.points)], axis=0)

#             if merged_points.shape[0] > target_size:
#                 merged_points = regularize_pc_point_count(merged_points, target_size)
#             merge_o3d = o3d.geometry.PointCloud()
#             merge_o3d.points = o3d.utility.Vector3dVector(merged_points)
#             # print("ori: ", len(acc_points.points))
#             # print("new1: ",len(merge_o3d.points))
#             # print("new2: ",len(merge_o3d.points))
#             merge_o3d, _ = merge_o3d.remove_statistical_outlier(
#                 # nb_neighbors=min(20, int(len(merge_o3d.points)/100)),
#                 nb_neighbors=20,
#                 std_ratio=1.0)
#             if (len(merge_o3d.points) / len(acc_points.points) < 0.2):
#                 return acc_points
#             else:
#                 return merge_o3d

#         obs_mask = np.zeros([ori_H, ori_W])
#         if self.target_pointcloud is None:
#             np.set_printoptions(threshold=np.inf)
#             obj_dict = {"center": np.zeros([0, 3]),
#                         "num_point": np.array([]),
#                         "mask_id": []}
#             for i, mask in enumerate(pred_visibles):
#                 valid_mask = np.logical_and(mask, depth_mask).flatten()
#                 object_pc = unorder_pc[valid_mask]
#                 pc_center = np.mean(object_pc, axis=0)
#                 pc_center = T[:3, :3].dot(pc_center) + T[:3, 3]
#                 out_range = False
#                 for axis in range(3):
#                     if (pc_center[axis] < self.bounds[axis][0]) or (pc_center[axis] > self.bounds[axis][1]):
#                         out_range = True
#                 if out_range: continue
#                 obs_mask += mask

#                 # if (pred_occs[i]) or (np.sum(mask) < 1000):
#                 #     continue
#                 if np.sum(mask) < 1000:
#                     continue
#                 mask = np.logical_and(mask, depth_mask).flatten()
#                 object_pc = unorder_pc[mask]

#                 if self.choose_mode == "2d":
#                     obj_dict["center"] = np.concatenate([obj_dict["center"], object_pc.mean(axis=0)[None]], axis=0)
#                 elif self.choose_mode == "3d":
#                     obj_dict["center"] = np.concatenate([obj_dict["center"], [pc_center]], axis=0)
#                 obj_dict["num_point"] = np.append(obj_dict["num_point"], len(object_pc))
#                 obj_dict["mask_id"].append(i)
                
                
#             print(len(obj_dict["mask_id"]))
#             if len(obj_dict["mask_id"]):
#                 if self.choose_mode == "2d":
#                     print("************use 2d")
#                     dist_index = np.argmin(obj_dict["center"][:, 2])
#                 elif self.choose_mode == "3d":
#                     print("**********use 3d")
#                     predefined_point = np.array([0., -0.6 , 0.])
#                     distances = [np.linalg.norm(center - predefined_point) for center in obj_dict["center"]]
#                     dist_index = np.argmin(distances)
#                     print("***********最近的中心：", obj_dict["center"][dist_index])
#                     print("*****************************************")
#                 # dist_index = np.argmin(np.sum((obj_dict["center"] - trans) ** 2, axis=1))
#                 # dist_index = np.argmax(obj_dict["num_point"])
#                 final_mask_target = pred_visibles[obj_dict["mask_id"][dist_index]]
#             else:
#                 # final_mask_target = None
#                 final_mask_target = pred_visibles[i]
            

#             print("3 time: {}".format(time.time() - start_time))
#             print(f"3self.target_pointcloud: {self.target_pointcloud}")
#         else:
#             """
#             project pointcloud back to plane
#             """
#             # previews_points = copy.deepcopy(self.target_pointcloud.points).T  # [3, N]
#             previews_points = regularize_pc_point_count(np.asarray(self.target_pointcloud.points), 1024).T
#             target_points_cam = T_inv[:3, :3].dot(previews_points) + T_inv[:3, [3]]
#             projected_target_pixel = self.intrinsic.dot(target_points_cam)
#             index = projected_target_pixel[2] > 0.03
#             projected_target_pixel = projected_target_pixel[:, index]
#             target_points_cam = target_points_cam[:, index]
#             pix_x, pix_y = (projected_target_pixel[0] / projected_target_pixel[2]).astype(np.int32), \
#                 (projected_target_pixel[1] / projected_target_pixel[2]).astype(np.int32)
#             valid_idx_mask = (pix_x > 0) * (pix_x < un_depth_img.shape[1] - 1) * \
#                 (pix_y > 0) * (pix_y < un_depth_img.shape[0] - 1)

#             def get_bbox(x, y):
#                 # bounding box
#                 x1 = np.min(x)
#                 x2 = np.max(x)
#                 y1 = np.min(y)
#                 y2 = np.max(y)
#                 area = (x2 - x1 + 1) * (y2 - y1 + 1)

#                 return (x1, x2, y1, y2), area

#             pix_bbox, pix_area = get_bbox(pix_x, pix_y)
#             final_idx = None
#             tmp_overlap = 0.
#             for i, mask in enumerate(pred_visibles):
#                 pc_center = np.mean(unorder_pc[mask.flatten()], axis=0)
#                 pc_center = T[:3, :3].dot(pc_center) + T[:3, 3]
#                 out_range = False
#                 for axis in range(3):
#                     if (pc_center[axis] < self.bounds[axis][0]) or (pc_center[axis] > self.bounds[axis][1]):
#                         out_range = True
#                 if out_range: continue

#                 obs_mask += mask

#                 mask_id = np.where(mask)
#                 mask_bbox, mask_area = get_bbox(mask_id[1], mask_id[0])
#                 xx1 = np.maximum(pix_bbox[0], mask_bbox[0])
#                 xx2 = np.minimum(pix_bbox[1], mask_bbox[1])
#                 yy1 = np.maximum(pix_bbox[2], mask_bbox[2])
#                 yy2 = np.minimum(pix_bbox[3], mask_bbox[3])

#                 w = np.maximum(0.0, xx2 - xx1 + 1)
#                 h = np.maximum(0.0, yy2 - yy1 + 1)
#                 inter = w * h
#                 ovr = inter / (pix_area + mask_area - inter)

#                 valid_depth = un_depth_img[pix_y[valid_idx_mask], pix_x[valid_idx_mask]]
#                 z = target_points_cam[2, valid_idx_mask]
#                 if len(valid_depth) and len(z):
#                     diff = np.mean(np.absolute(valid_depth - z))
#                     print('mean depth diff', diff)
#                 else:
#                     diff = 0.

#                 if (ovr < 0.3) or (diff > 0.15):
#                     continue
#                 else:
#                     tmp_overlap = ovr if ovr > tmp_overlap else tmp_overlap
#                     final_idx = i

#             if final_idx is not None:
#                 final_mask_target = pred_visibles[final_idx]
#             else:
#                 final_mask_target = None
#             print("target:", np.sum(final_mask_target))

#         if isinstance(final_mask_target, np.ndarray):
#             final_mask_target = final_mask_target.astype(np.uint8)
#             kernel = np.ones((5, 5), np.uint8)
#             print(f"len(pred_visibles): {len(pred_visibles)}")
#             if len(pred_visibles) > 1:
#                 table_mask = np.logical_not(obs_mask).astype(np.uint8)
#                 obs_mask[np.where(final_mask_target > 0)] = 0
#                 obs_mask = obs_mask.clip(0, 1).astype(np.uint8)
#                 print("target:", np.sum(final_mask_target))
#                 # final_mask_target, table_mask, obs_mask = [cv2.erode(m, kernel, iterations=3).astype(np.bool_) \
#                 #                             for m in [final_mask_target, table_mask, obs_mask]]
#                 final_mask_target = final_mask_target.astype(np.bool_)
#                 table_mask = table_mask.astype(np.bool_)
#                 obs_mask = obs_mask.astype(np.bool_)
#                 sampled_pc = backproject_pc(final_mask_target, frame=frame)
#                 print(np.sum(obs_mask), np.sum(final_mask_target))
#                 print(f"sampled_pc: {np.asarray(sampled_pc.points).shape}")
#                 # plt.imsave("/home/user/uoais_ws/test_frame.png", obs_mask)
#                 if np.sum(obs_mask) > 1000:
#                     scene_pc = ratio_pc_concat(obs_mask, table_mask, frame=frame)
#                     # scene_pc = backproject_pc(np.logical_not(final_mask_target))
#                 else:
#                     # scene_pc = backproject_pc(np.logical_not(final_mask_target))
#                     scene_pc = backproject_pc(table_mask, frame=frame)
#             else:
#                 final_mask_scene = np.logical_not(final_mask_target).astype(np.uint8)
#                 final_mask_target = cv2.erode(final_mask_target, kernel, iterations=3).astype(np.bool8)
#                 final_mask_scene = cv2.erode(final_mask_scene, kernel, iterations=3).astype(np.bool8)

#                 sampled_pc, scene_pc = [backproject_pc(m, frame=frame) for m in [final_mask_target, final_mask_scene]]

#             print("4 time: {}".format(time.time() - start_time))
#             print(f"4self.target_pointcloud: {self.target_pointcloud}")
            
#             if self.target_pointcloud is None:
#                 # visualize part
#                 if self.vis:
#                     axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
#                     o3d.visualization.draw_geometries([sampled_pc, axes])
#                 self.target_pointcloud = sampled_pc.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)[0]
#                 self.scene_pointcloud = scene_pc.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)[0]
#             else:
#                 # visualize part
#                 if self.vis:
#                     self.target_pointcloud.paint_uniform_color([1, 0, 0])
#                     sampled_pc.paint_uniform_color([0, 1, 0])
#                     axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
#                     o3d.visualization.draw_geometries([self.target_pointcloud, sampled_pc, axes])
#                 print(sampled_pc, scene_pc)
#                 self.target_pointcloud = accu_point_cloud(sampled_pc, self.target_pointcloud)
#                 self.scene_pointcloud = accu_point_cloud(scene_pc, self.scene_pointcloud)
#             print("5 time: {}".format(time.time() - start_time))
#             print(f"5self.target_pointcloud: {self.target_pointcloud}")

#             if self.vis:
#                 axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
#                 o3d.visualization.draw_geometries([self.scene_pointcloud, axes])

#         header = Header()
#         header.stamp = rospy.Time().now()
#         header.frame_id = "base"
#         target_point = regularize_pc_point_count(np.asarray(self.target_pointcloud.points), 1024, use_farthest_point=True)
#         scene_point = regularize_pc_point_count(np.asarray(self.scene_pointcloud.points), 1024, use_farthest_point=True)
#         print(f"scene_point: {len(scene_point)}")
#         pc_target = create_cloud_xyz32(header, np.asarray(target_point))
#         pc_scene = create_cloud_xyz32(header, np.asarray(scene_point))
#         self.points_pub.publish(pc_target)
#         self.obs_points_pub.publish(pc_scene)

#         end_time = time.time()
#         print("***********target_pointcloud:", self.target_pointcloud)
#         print("total time: {}".format(end_time - start_time))
        
        
#         self.target_pointcloud = None # This is for repeat rostopic start, delte if pc needs combining
#         # return results
#         return pc_target, pc_scene



# def regularize_pc_point_count(pc, npoints, use_farthest_point=True):
#     """
#     If point cloud pc has less points than npoints, it oversamples.
#     Otherwise, it downsample the input pc to have npoint points.
#     use_farthest_point: indicates whether to use farthest point sampling
#     to downsample the points. Farthest point sampling version runs slower.
#     """
#     if pc.shape[0] > npoints:
#         if use_farthest_point:
#             pc = torch.from_numpy(pc).cuda()[None].float()
#             new_xyz = (
#                 gather_operation(
#                     pc.transpose(1, 2).contiguous(), furthest_point_sample(pc[..., :3].contiguous(), npoints)
#                 )
#                 .contiguous()
#                 )
#             pc = new_xyz[0].T.detach().cpu().numpy()

#         else:
#             center_indexes = np.random.choice(
#                 range(pc.shape[0]), size=npoints, replace=False
#             )
#             pc = pc[center_indexes, :]
#     else:
#         required = npoints - pc.shape[0]
#         print(f"pc.shape[0]: {pc.shape[0]}")
#         if required > 0:
#             index = np.random.choice(range(pc.shape[0]), size=required)
#             pc = np.concatenate((pc, pc[index, :]), axis=0)

#     return pc


# if __name__ == '__main__':
#     print("launch uoais point cloud preprocess")
#     uoais = UOAIS()
#     rospy.spin()
