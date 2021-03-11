#!/usr/bin/env python3

import rospy
from rospkg import RosPack
# import tf2_ros
from std_msgs.msg import Header
from mesh_partition.msg import MeshEnvironment as envMsg
from mesh_partition.msg import PerchTarget as pchMsg
import geometry_msgs.msg as geoMsg
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

from scripts.optimize_ros import optimize
# from scripts.optimize_ros import optimize
from geom.geometry3d import eul_to_qt
from tools.tools import make_path_absolute
import configparser


class PerchPicker:
    def __init__(self, part_mesh_topic='part_mesh_env', perch_topic='perch_targets'):
        rospy.init_node('perch_picker', anonymous=True)

        self.pub = rospy.Publisher(perch_topic, pchMsg, queue_size=100)
        # self.gz_pub = rospy.Publisher(perch_topic, geoMsg, queue_size=100)
        # self.tf_pub = rospy.Publisher(perch_topic, geoMsg, queue_size=100)
        self.sub = rospy.Subscriber(part_mesh_topic, envMsg, self.roi_callback)

    def roi_callback(self, msg):
        rospy.loginfo(rospy.get_caller_id() + " Mesh filepaths received. Starting Camera Placement Optimization.")

        # launch optimization from here
        best_placements = optimize(seg_env_prototype=make_path_absolute(msg.surf_path_prototype),
                                   target_prototype=make_path_absolute(msg.target_path_prototype),
                                   cluster_env_path=make_path_absolute(msg.clustered_env_path),
                                   full_env_path=make_path_absolute(msg.full_env_path),
                                   enable_user_confirmation=True)

        rospy.loginfo(" Publishing Optimal Placements...")

        count = 0

        for bp in best_placements:
            rospy.loginfo(" Publishing Camera " + str(count))
            pch_msg = pchMsg()
            state_msg = ModelState()
            static_transform = geoMsg.TransformStamped()
            pos_msgs = [pch_msg.pose.position, state_msg.pose.position, static_transform.transform.translation]
            rot_msgs = [pch_msg.pose.orientation, state_msg.pose.orientation, static_transform.transform.rotation]
            h = Header()
            h.stamp = rospy.Time.now()
            q = eul_to_qt(bp.pose[3:])

            cam_name = "iris_"+str(count+1)
            frame_name = "iris_"+str(count+1)+"/camera__optical_center_link"

            for i in range(len(pos_msgs)):
                pos_msgs[i].x = bp.pose[0]
                pos_msgs[i].y = bp.pose[1]
                pos_msgs[i].z = bp.pose[2]
                rot_msgs[i].x = q[0]
                rot_msgs[i].y = q[1]
                rot_msgs[i].z = q[2]
                rot_msgs[i].w = q[3]

            pch_msg.header = h
            pch_msg.camera_id = bp.camera_id
            # pch_msg.pose.position.x = bp.pose[0]
            # pch_msg.pose.position.y = bp.pose[1]
            # pch_msg.pose.position.z = bp.pose[2]
            # pch_msg.pose.orientation.x = q[0]
            # pch_msg.pose.orientation.y = q[1]
            # pch_msg.pose.orientation.z = q[2]
            # pch_msg.pose.orientation.w = q[3]
            self.pub.publish(pch_msg)

            state_msg = ModelState()
            state_msg.model_name = cam_name

            # state_msg.pose.position.x = bp.pose[0]
            # state_msg.pose.position.y = bp.pose[1]
            # state_msg.pose.position.z = bp.pose[2]
            # state_msg.pose.orientation.x = q[0]
            # state_msg.pose.orientation.y = q[1]
            # state_msg.pose.orientation.z = q[2]
            # state_msg.pose.orientation.w = q[3]
            # TODO: RE-ADD
            # rospy.wait_for_service('/gazebo/set_model_state')
            print("Sending Gazebo State Message: ")
            print(str(state_msg))
            # try:
            #     set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            #     resp = set_state(state_msg)
            #     print(str(resp))
            # except rospy.ServiceException as e:
            #     print("Service call failed: %s" % e)

            static_transform = geoMsg.TransformStamped()
            static_transform.header.stamp = rospy.Time.now()
            static_transform.header.frame_id = "world"
            static_transform.child_frame_id = frame_name

            # static_transform.transform.translation.x = bp.pose[0]
            # static_transform.transform.translation.y = bp.pose[1]
            # static_transform.transform.translation.z = bp.pose[2]
            # static_transform.transform.rotation.x = q[0]
            # static_transform.transform.rotation.y = q[1]
            # static_transform.transform.rotation.z = q[2]
            # static_transform.transform.rotation.w = q[3]

            # broadcaster = tf2_ros.StaticTransformBroadcaster()
            # broadcaster.sendTransform(static_transform)

            count += 1

        rospy.loginfo(" CPO Complete")
        rospy.loginfo(" Initializing Gazebo Simulation...")


if __name__ == '__main__':

    rospy.loginfo("Where to Perch Node starting...")

    config = configparser.ConfigParser()
    rp = RosPack()
    config.read(rp.get_path(name="perch_placement")+"/src/config/opt.ini")

    part_mesh_path_topic = config['ROS']['part_mesh_path_topic']
    perch_loc_topic = config['ROS']['perch_loc_topic']

    try:
        pp = PerchPicker(part_mesh_topic=part_mesh_path_topic, perch_topic=perch_loc_topic)
        rospy.loginfo("Where to Perch Node is live! Expecting messages on... " + part_mesh_path_topic)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
