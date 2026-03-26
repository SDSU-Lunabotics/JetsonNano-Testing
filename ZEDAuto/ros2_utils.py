try:
    import rclpy
    from sensor_msgs.msg import PointCloud2, PointField
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Header

    HAS_ROS2 = True
except Exception:
    HAS_ROS2 = False


def setup_ros2(enable_ros2):
    node = None
    pc_pub = None
    pc_fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    if enable_ros2:
        if not HAS_ROS2:
            print("ROS2 Python libs not found. Did you source ROS2 and install rclpy?")
        else:
            rclpy.init()
            node = rclpy.create_node("zed_ground_wall")
            pc_pub = node.create_publisher(PointCloud2, "zed/pointcloud", 10)
            print("ROS2 enabled: publishing /zed/pointcloud")
    else:
        print("ROS2 disabled (run with --ros2 to publish /zed/pointcloud)")
    return node, pc_pub, pc_fields


def publish_pointcloud(node, pc_pub, pc_fields, xyz, frame_id):
    if node is None or pc_pub is None:
        return
    header = Header()
    header.stamp = node.get_clock().now().to_msg()
    header.frame_id = frame_id
    pts = xyz.astype("float32")
    msg = point_cloud2.create_cloud(header, pc_fields, pts.tolist())
    pc_pub.publish(msg)
    rclpy.spin_once(node, timeout_sec=0.0)


def shutdown_ros2(node):
    if node is None or not HAS_ROS2:
        return
    node.destroy_node()
    rclpy.shutdown()
