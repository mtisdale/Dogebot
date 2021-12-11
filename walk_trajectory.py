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
        self.kin_2 = kin.Kinematics(robot, 'body', 'tip_2') 
        self.kin_3 = kin.Kinematics(robot, 'body', 'tip_3') 
        self.kin_4 = kin.Kinematics(robot, 'body', 'tip_4')     
        self.kin = [self.kin_1, self.kin_2, self.kin_3, self.kin_4]
        self.stride_freq = 1.0
        self.stride_ht = 0.3
        self.legs = [0, 1, 2, 3]                                        # Order of leg cycling
        delta_forward = np.array([0.5, 0.0, 0.0]).reshape((3,1))        # For determining xf
        delta_backward = np.array([-0.5, 0.0, 0.0]).reshape((3,1))      # For determining xf
        
        
        # Initialize the feet positions 
        theta_0 = np.array([0.0, 0.0, 0.0]).reshape((3,1))
        x1 = np.array([1.0, 0.5, 0.0]).reshape((3,1))       # Starting foot 1 placement 
        x2 = np.array([1.0, -0.5, 0.0]).reshape((3,1))       # Starting foot 2 placement
        x3 = np.array([-1.0, 0.5, 0.0]).reshape((3,1))       # Starting foot 3 placement
        x4 = np.array([-1.0, -0.5, 0.0]).reshape((3,1))       # Starting foot 4 placement
                
        
        # Instantiate update function variables
        self.index = 0
        self.t0    = 0.0
        self.t1 = 0.0
        self.t_prev = 0.0
        self.lamb = 0.5
        self.reset_guess = theta_0       
        self.q_prev = [theta_0, theta_0, theta_0, theta_0]       
        self.count_steps = 0
                           
        self.xi = [x1, x2, x3, x4]                              
        self.xf = [0, 0, 0, 0]
        self.xf2 = [0, 0, 0, 0]
        self.x_curr = [x1, x2, x3, x4]      
        # self.v = 1.0 / (4*len(self.legs))
        self.v = 0.08
        for i in range(len(self.legs)):
            self.xf[i] = self.xi[i] + delta_forward
            self.xf2[i] = self.xf[i] + delta_forward
            
       
       
       # Instantiate the segments 
        self.segments = (Goto5(0.0, 2.0, self.stride_freq, 'Path', self.legs[0]),                           # For the upwards parabolic movement
                         Goto5(0.0, 2.0, self.stride_freq, 'Path', self.legs[1]),                           # For the upwards parabolic movement
                         Goto5(0.0, 2.0, self.stride_freq, 'Path', self.legs[2]),                           # For the upwards parabolic movement  
                         Goto5(0.0, 2.0, self.stride_freq, 'Path', self.legs[3]),                           # For the upwards parabolic movement
                         Goto5(0.0, 2.0, self.stride_freq, 'Path', self.legs[0]),                           # For the upwards parabolic movement
                         Goto5(0.0, 2.0, self.stride_freq, 'Path', self.legs[1]),                           # For the upwards parabolic movement
                         Goto5(0.0, 2.0, self.stride_freq, 'Path', self.legs[2]),                           # For the upwards parabolic movement
                         Goto5(0.0, 2.0, self.stride_freq, 'Path', self.legs[3]))                           # For the upwards parabolic movement

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

            # Update position of leg just placed on the ground
            last_leg = self.segments[self.index].leg()
            phase = int(self.count_steps / len(self.legs)) % len(self.legs)
            if (phase == 3):
                self.x_curr[last_leg] = self.xi[last_leg]    
            elif (phase == 2):
                self.x_curr[last_leg] = self.xf[last_leg]
            elif (phase == 1):
                self.x_curr[last_leg] = self.xf2[last_leg]
            else:
                self.x_curr[last_leg] = self.xf[last_leg]                
            self.count_steps += 1
            
            # If the list were cyclic, you could go back to the start with
            self.index = (self.index+1) % len(self.segments)   
            if (self.index == 0):
                self.t1 = self.t0
            
        # Check whether we are done with all segments
        if (self.index >= len(self.segments)):
            rospy.signal_shutdown("Done with motion")
            return
        
        # Implementation for the different splines    
        leg = self.segments[self.index].leg()
        phase = int(self.count_steps / len(self.legs)) % len(self.legs)
        q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        qdot = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Determine transforms of body, for now fixed
        # I NEED TO FIGURE OUT THIS EXACTLY FOR A CONTINUOUS LOOP
        if (phase == 2 or phase == 3):
            x_pos = -self.v*(t-self.t1) + self.t1*self.v
            p_body = np.array([x_pos, 0.0, 1.0]).reshape((3,1))
            v_body = np.array([-self.v, 0.0, 0.0]).reshape((3,1))
        else:
            x_pos = self.v*(t-self.t1)
            p_body = np.array([x_pos, 0.0, 1.0]).reshape((3,1))
            v_body = np.array([self.v, 0.0, 0.0]).reshape((3,1))
        R_body = Rz(0.0)
        w_body = np.array([0.0, 0.0, 0.0]).reshape((3,1))       
        T_body = T_from_Rp(R_body, p_body)
        
        
        if (self.segments[self.index].space() == 'Path'):
            for i in range(len(self.legs)):
                if (i == leg):  
                # Perform the path spline here for the lifting leg
                    (s, sdot) = self.segments[self.index].evaluate(t-self.t0)
                    
                    # Choose the start and end position                      
                    if (phase == 3):
                        xi = self.xf[leg]
                        xf = self.xi[leg]    
                    elif (phase == 2):
                        xi = self.xf2[leg]
                        xf = self.xf[leg]
                    elif (phase == 1):
                        xi = self.xf[leg]
                        xf = self.xf2[leg]  
                    else:
                        xi = self.xi[leg]
                        xf = self.xf[leg]
                    
                    # Compute errors and Jacobian from equations
                    (T, J) = self.kin[leg].fkin(self.q_prev[leg])                                   # T and J are in body space
                    x_des = self.pd(s, xi, xf)                                                      # World space
                    xdot_world = self.vd(s, sdot, xi, xf)                                           # World space
                    x_tilda = self.ep(R_body.T @ (x_des - p_body), p_from_T(T))
                    xdot = R_body.T @ (xdot_world - v_body - np.cross(w_body, x_des-p_body, axis=0))
                    
                    # Compute q and qdot using Velocity IKIN Equations
                    xvec = xdot + self.lamb*x_tilda
                    qdot_vals = np.linalg.pinv(J[0:3,:]) @ xvec
                    dt = t-self.t_prev
                    q_vals = self.q_prev[leg] + dt*qdot_vals
                    
                    # Update values to use in next iteration
                    self.q_prev[leg] = q_vals
                    self.t_prev = t
                    
                    # Save to general q and qdot list to send to jntmsg later
                    for j in range(3):
                        q[3*leg+j] = q_vals[j]
                        qdot[3*leg+j] = qdot_vals[j]
                    
                else:       
                # Compute new joint positions for all other legs given fixed position, use i
                    feet_pos = self.x_curr[i]
                    feet_vel = np.array([0.0, 0.0, 0.0]).reshape((3,1))
                    pos_prime = R_body.T @ (feet_pos - p_body)                                              # Foot Position in body space     
                    q_vals = self.kin[i].ikin(pos_prime, self.q_prev[i])                                    # Body Space IKIN
                    (T, J) = self.kin[i].fkin(q_vals)                                                       # Both T and J are in body space
                    vel_prime = R_body.T @ (feet_vel - v_body - np.cross(w_body, feet_pos-p_body, axis=0))  # Foot Velocity in body space
                    qdot_vals = np.linalg.pinv(J[0:3,:]) @ vel_prime
                    
                    # Update values to use in next iteration
                    self.q_prev[i] = q_vals
                    
                    # Save to general q and qdot list to send to jntmsg later
                    for j in range(3):
                        q[3*i+j] = q_vals[j]
                        qdot[3*i+j] = qdot_vals[j]
                    
                    
                    
        # Apply the computed pos/vel into all joint messages
        for k in range(len(self.jointdict)):  # 3 DOFs per leg
            self.jntmsg.position[k] = q[k]
            self.jntmsg.velocity[k] = qdot[k]                     
        '''
        elif (self.segments[self.index].space() == 'Task'):
            leg = self.segments[self.index].leg()
            (feet_pos, feet_vel) = self.segments[self.index].evaluate(t-self.t0)
            pos_prime = R_body.T @ (feet_pos-p_body)                                                    # Foot Position in body space
            q = self.kin[leg].ikin(pos_prime, self.q_prev[leg])                                         # IKIN wrt body space
            
            (T, J) = self.kin[leg].fkin(q)                                                              # T and J are both in body space
            vel_prime = R_body.T @ (feet_vel - v_body - np.cross(w_body, (feet_pos-p_body), axis=0))    # Foot Velocity in body space
            qdot = np.linalg.pinv(J[0:3,:]) @ vel_prime                  
            self.q_prev[leg] = q
        '''
          
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
