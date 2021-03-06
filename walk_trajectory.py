#!/usr/bin/env python3
#
#   Publish:   /joint_states   sensor_msgs/JointState
#
import rospy
import math
import tf2_ros
import numpy as np
import kinematics as kin

from urdf_parser_py.urdf import Robot
from sensor_msgs.msg     import JointState
from geometry_msgs.msg   import Vector3, Quaternion, Transform, TransformStamped
from std_msgs.msg        import Bool

from splines import CubicSpline, Goto, Hold, Stay, QuinticSpline, Goto5
from kinematics import p_from_T, q_from_T, T_from_Rp, Rx, Ry, Rz


#
#  Transform Helper Function
#
#  Convert a numpy 4x4 transform into a ROS transform message.
#
def transform_from_T(T):
    # Instantiate the transform
    trans = Transform()

    # Set the positions.
    p = p_from_T(T)
    trans.translation.x = p[0,0]
    trans.translation.y = p[1,0]
    trans.translation.z = p[2,0]

    # Set the quaternions.  Note, ROS transforms place the w component
    # last, while our kinematics library places the w component first.
    quat = q_from_T(T)
    trans.rotation.x = quat[1]
    trans.rotation.y = quat[2]
    trans.rotation.z = quat[3]
    trans.rotation.w = quat[0]

    # Return the transform
    return trans

#
#  Generator Class
#
class Generator:
    # Initialize.
    def __init__(self):
        # Grab the robot's URDF from the parameter server to find the
        # joints names and the root link name.
        robot = Robot.from_parameter_server()

        # Build up a dictionary of the joint names.  
        self.jointdict = {'hip_1_z': 0, 'hip_1_x': 1, 'knee_1': 2,
                            'hip_2_z': 3, 'hip_2_x': 4, 'knee_2': 5,
                            'hip_3_z': 6, 'hip_3_x': 7, 'knee_3': 8,
                            'hip_4_z': 9, 'hip_4_x': 10, 'knee_4': 11}
#        self.jointdict = {}
#        for joint in robot.joints:
#            if ((joint.type == 'continuous') or
#                (joint.type == 'revolute') or
#                (joint.type == 'prismatic')):
#                n = len(self.jointdict)
#                self.jointdict[joint.name] = n
#                rospy.loginfo("Joint %2d: '%s'", n, joint.name)
#            elif (joint.type != 'fixed'):
#                # There shouldn't be any other types...
#                rospy.logwarn("Joint '%s' has unknown type %s", joint.name, joint.type)

        # Grab the URDF root link name. This is the link "body" of our URDF
        root = robot.joints[0].parent
        print()
        print(root)
        print()
        rospy.loginfo("Root link '%s'", root)

        # Instantiate a joint state message.  Pre-populate the joint
        # names and simultaneously clear the default commands.
        self.jntmsg          = JointState()
        self.jntmsg.name     = list(self.jointdict.keys())
        self.jntmsg.position = [0.0] * len(self.jointdict)
        self.jntmsg.velocity = [0.0] * len(self.jointdict)

        # Instantiate a transform message to set the URDF's root
        # transform.  Each transform has a parent (defined in the
        # header) and child frame.  This connects 'world' to the root.
        self.tfmsg = TransformStamped()
        self.tfmsg.header.frame_id = 'world'
        self.tfmsg.child_frame_id  = root

        # Create a publisher to send the joint commands and a TF2
        # transform broadcaster to send the root link transform.
        self.pub = rospy.Publisher("/joint_states", JointState, queue_size=10)
        self.brd = tf2_ros.TransformBroadcaster()
        # Add some time to make the connections.  This isn't strictly
        # necessary, but avoids sending initial messages into a void.
        rospy.sleep(0.25)
        
        rospy.Subscriber('/boolean', Bool, self.receive_Bool)

        # Instantiate the Kinematics
        self.kin_1 = kin.Kinematics(robot, 'body', 'tip_1') 
        self.kin_2 = kin.Kinematics(robot, 'body', 'tip_2') 
        self.kin_3 = kin.Kinematics(robot, 'body', 'tip_3') 
        self.kin_4 = kin.Kinematics(robot, 'body', 'tip_4')            
        
        
        # Initialize the segment positions MAKE INTO A BETTER ARRAY FOR ALL FOUR FEET LATER
        theta_guess = np.array([0.0, 0.0, 0.0]).reshape((3,1))
        x1 = np.array([1.0, 0.5, 0.0]).reshape((3,1))       # Starting foot 1 placement 
        x2 = np.array([1.75, 0.5, 0.0]).reshape((3,1))       # Ending foot 1 placement 
        q1 = self.kin_1.ikin(x1, theta_guess)
        q2 = self.kin_1.ikin(x2, theta_guess)
        
        self.segments = (Goto(q1, q2, 3.0, 'Joint'),        # For the upwards parabolic movement
                         Goto(x2, x1, 3.0, 'Task'))         # For movement along the ground
        self.index = 0
        self.t0    = 0.0
        self.last_guess = theta_guess



    def receive_Bool(self, msg):
        print(msg.data)
        
    # Update every 10ms!
    def update(self, t):


        # If the current segment is done, shift to the next.
        if (t - self.t0 >= self.segments[self.index].duration()):
            self.t0    = self.t0 + self.segments[self.index].duration()
            # self.index = (self.index+1)
            # If the list were cyclic, you could go back to the start with
            self.index = (self.index+1) % len(self.segments)
            # self.last_guess = self.reset_guess 
            
        # Check whether we are done with all segments
        if (self.index >= len(self.segments)):
            rospy.signal_shutdown("Done with motion")
            return
            
        
        # Implementation for the different splines    
        if (self.segments[self.index].space() == 'Joint'):
            # Joint space spline actually causes it to drag across the ground... Need to have secondary task maybe?
            (position, velocity) = self.segments[self.index].evaluate(t-self.t0)
            
        elif (self.segments[self.index].space() == 'Task'):
            # Test to see if this is even gonna work lmao
            (cart_position, cart_velocity) = self.segments[self.index].evaluate(t-self.t0)
            position = self.kin_1.ikin(cart_position, self.last_guess)
            (T, J) = self.kin_1.fkin(position)
            velocity = np.linalg.inv(J[0:3,0:3]) @ cart_velocity
            self.last_guess = position
            
            
        # Apply the computed pos/vel into joint message
        i = self.jointdict['hip_1_z']
        j = self.jointdict['hip_1_x']
        k = self.jointdict['knee_1']
        self.jntmsg.position[i] = position[0]
        self.jntmsg.position[j] = position[1]
        self.jntmsg.position[k] = position[2]
        self.jntmsg.velocity[i] = velocity[0]
        self.jntmsg.velocity[j] = velocity[1]
        self.jntmsg.velocity[k] = velocity[2]

        # For a fixed body, maybe depends on inputs here
        R_init = Rz(0.0)
        p_init = np.array([0.0, 0.0, 1.0]).reshape((3,1))
        T_init = T_from_Rp(R_init, p_init)
        
        # Set the root link transform
        self.tfmsg.transform = transform_from_T(T_init)

        # Send the joint command and root link transform (with the current time).
        timestamp = rospy.Time.now()
        
        self.jntmsg.header.stamp = timestamp
        self.pub.publish(self.jntmsg)

        self.tfmsg.header.stamp  = timestamp
        self.brd.sendTransform(self.tfmsg)
        
        
        
#
#  Main Code
#
if __name__ == "__main__":
    # Prepare/initialize this node.
    rospy.init_node('generator')

    # Instantiate the trajectory generator object, encapsulating all
    # the computation and local variables.
    generator = Generator()

    # Prepare a servo loop at 100Hz.
    rate  = 100;
    servo = rospy.Rate(rate)
    dt    = servo.sleep_dur.to_sec()
    rospy.loginfo("Running the servo loop with dt of %f seconds (%fHz)" %
                  (dt, rate))


    # Run the servo loop until shutdown (killed or ctrl-C'ed).
    starttime = rospy.Time.now()
    while not rospy.is_shutdown():

        # Current time (since start)
        servotime = rospy.Time.now()
        t = (servotime - starttime).to_sec()

        # Update the controller.
        generator.update(t)

        # Wait for the next turn.  The timing is determined by the
        # above definition of servo.
        servo.sleep()
