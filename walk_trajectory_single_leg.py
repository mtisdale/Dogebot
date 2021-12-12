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

from splines import CubicSpline, Goto, Hold, Stay, QuinticSpline, Goto5
from kinematics import p_from_T, q_from_T, R_from_T, T_from_Rp, Rx, Ry, Rz


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
#
#   Initialization Function
#
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


        # Instantiate the Kinematics
        self.kin_1 = kin.Kinematics(robot, 'body', 'tip_1') 
        self.stride_freq = 3.0
        self.stride_ht = 0.3
        delta_forward = np.array([0.5, 0.0, 0.0]).reshape((3,1))        # For determining xf
        delta_backward = np.array([-0.5, 0.0, 0.0]).reshape((3,1))      # For determining xf
        
        
        # Initialize the feet positions 
        theta_guess = np.array([0.0, 0.0, 0.0]).reshape((3,1))
        self.xi = np.array([1.0, 0.5, 0.0]).reshape((3,1))       # Starting foot 1 placement 
        self.xf = self.xi + delta_forward      
        
        # Instantiate update function variables
        self.index = 0
        self.t0    = 0.0
        self.t_prev = 0.0
        self.lamb = 0.5              
        self.last_guess = theta_guess
        self.reset_guess = theta_guess
        self.q_prev = theta_guess           
        
        self.segments = (Hold(self.xi, 1.0, 'Task', 0),
                        #Goto5(self.xi, self.xf, self.stride_freq, 'Task', 0),
                        Goto5(0.0, 2.0, self.stride_freq, 'Path', 0),
                        Goto5(self.xf, self.xi, self.stride_freq, 'Task', 0))

#
#   Path Spline Functions
#
    def pd(self, s, xi, xf):
        midx = 0.5*(xf[0,0]-xi[0,0])
        midy = 0.5*(xf[1,0]-xi[1,0])
        if s<=1:
            return np.array([midx*s+xi[0,0], midy*s+xi[1,0], self.stride_ht*s]).reshape((3,1))
        else:
            return np.array([midx*s+xi[0,0], midy*s+xi[1,0], self.stride_ht*(2-s)]).reshape((3,1))
    
    def vd(self, s, sdot, xi, xf):
        midx = 0.5*(xf[0,0]-xi[0,0])
        midy = 0.5*(xf[1,0]-xi[1,0])
        if s<=1:
            return np.array([midx*sdot, midy*sdot, self.stride_ht*sdot]).reshape((3,1))
        else:
            return np.array([midx*sdot, midy*sdot, -self.stride_ht*sdot]).reshape((3,1))
        
    def ep(self, pd, p):
        return (pd-p)

#
#   Update Function
#
    def update(self, t):

        # If the current segment is done, shift to the next.
        if (t - self.t0 >= self.segments[self.index].duration()):
            self.t0    = self.t0 + self.segments[self.index].duration()
            # self.index = (self.index+1)
            # If the list were cyclic, you could go back to the start with
            print('Next Segment. \n')
            self.index = (self.index+1) % len(self.segments)
            self.q_prev = self.reset_guess
            self.last_guess = self.reset_guess
            
        # Check whether we are done with all segments
        if (self.index >= len(self.segments)):
            rospy.signal_shutdown("Done with motion")
            return
        
        
        # Determine transforms of body, for now fixed
        R_body = Rz(0.0)
        p_body = np.array([0.0, 0.0, 1.0]).reshape((3,1))
        v_body = np.array([0.0, 0.0, 0.0]).reshape((3,1))
        w_body = np.array([0.0, 0.0, 0.0]).reshape((3,1))
        T_body = T_from_Rp(R_body, p_body)
        
        
        # Implementation for the different splines
        if (self.segments[self.index].space() == 'Path'):
            (s, sdot) = self.segments[self.index].evaluate(t-self.t0)
                        
            # Compute errors and Jacobian from equations
            (T, J) = self.kin_1.fkin(self.q_prev)                               # T and J are in body space
            x_des = self.pd(s, self.xi, self.xf)                                # World space
            xdot_world = self.vd(s, sdot, self.xi, self.xf)                     # World space
            x_tilda = self.ep(R_body.T @ (x_des - p_body), p_from_T(T))
            xdot = R_body.T @ (xdot_world - v_body - np.cross(w_body, x_des-p_body, axis=0))
            
            # Compute q and qdot using Velocity IKIN Equations
            xvec = xdot + self.lamb*x_tilda
            qdot = np.linalg.pinv(J[0:3,:]) @ xvec
            dt = t-self.t_prev
            q = self.q_prev + dt*qdot
            
            # Update values to use in next iteration
            self.q_prev = q
            self.t_prev = t
            
        elif (self.segments[self.index].space() == 'Task'):
            (feet_pos, feet_vel) = self.segments[self.index].evaluate(t-self.t0)
            pos_prime = R_body.T @ (feet_pos-p_body)                                                    # Foot Position in body space
            q = self.kin_1.ikin(pos_prime, self.q_prev)                                          # IKIN is with respect to body space
            (T, J) = self.kin_1.fkin(q)                                                              # T and J are both in body space
            vel_prime = R_body.T @ (feet_vel-v_body-np.cross(w_body, (feet_pos-p_body), axis=0))    # Foot Velocity in body space
            qdot = np.linalg.inv(J[0:3,:]) @ vel_prime      
            # Change below last guess to account for all legs later                
            self.q_prev = q
        
        
        # Apply the computed pos/vel into joint messages
        for i in range(3):  # 3 DOFs per leg
            self.jntmsg.position[i] = q[i]
            self.jntmsg.velocity[i] = qdot[i]
        
          
        # Set the root link transform
        self.tfmsg.transform = transform_from_T(T_body)

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
