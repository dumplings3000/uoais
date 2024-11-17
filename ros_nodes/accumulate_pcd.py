#!/usr/bin/env python3
import rospy
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

pointcloud_accumulation = o3d.geometry.PointCloud()

def pointcloud_callback(pointcloud_msg):
    # 將PointCloud2消息轉換為點雲數據
    points = pc2.read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True)
    point_list = list(points)

    # 創建Open3D點雲對象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_list)

    # 累加點雲數據
    pointcloud_accumulation += pcd

    # 顯示點雲
    o3d.visualization.draw_geometries([pointcloud_accumulation])

def listener():
    rospy.init_node('pointcloud_listener', anonymous=True)
    rospy.Subscriber("/uoais/target_pcd", PointCloud2, pointcloud_callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
