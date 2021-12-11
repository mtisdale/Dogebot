#!/usr/bin/env python3
#
#   inverse_kinematics.py
#
#   Kinematics Class and Helper Functions (HW 4.5)
#
#   This computes the inverse kinematics and Jacobian using the
#   kinematic chain.  It also includes test code when run
#   independently.
#
import rospy
import numpy as np
import math

from urdf_parser_py.urdf import Robot


#
#  Kinematics Helper Functions
#
#  These compute
#    3x1 point vectors "p"
#    3x1 axes of rotation "e"
#    3x3 rotation matrices "R"
#    1x4 quaternions "q"
#    4x4 transforms "T"
#  from each other or
#    1x3 xyz vector of positions
#    1x3 rpy vector of angles
#    1x6 xyz/rpy origin
#    1x3 axis vector
#
# Points:
def p_from_T(T):
    return T[0:3,3:4]

def p_from_xyz(xyz):
    return np.array(xyz).reshape((3,1))

# Axes:
def e_from_axis(axis):
    return np.array(axis).reshape((3,1))

# Rotation Matrices:
def R_from_T(T):
    return T[0:3,0:3]

def R_from_rpy(rpy):
    return Rz(rpy[2]) @ Ry(rpy[1]) @ Rx(rpy[0])

def Rx(theta):
    c = np.cos(theta);
    s = np.sin(theta);
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
def Ry(theta):
    c = np.cos(theta);
    s = np.sin(theta);
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
def Rz(theta):
    c = np.cos(theta);
    s = np.sin(theta);
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def R_from_axisangle(axis, theta):
    ex = np.array([[     0.0, -axis[2],  axis[1]],
                   [ axis[2],      0.0, -axis[0]],
                   [-axis[1],  axis[0],     0.0]])
    return np.eye(3) + np.sin(theta) * ex + (1.0-np.cos(theta)) * ex @ ex

def R_from_q(q):
    norm2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]
    return - np.eye(3) + (2/norm2) * (
      np.array([[q[1]*q[1]+q[0]*q[0],q[1]*q[2]-q[0]*q[3],q[1]*q[3]+q[0]*q[2]],
                [q[2]*q[1]+q[0]*q[3],q[2]*q[2]+q[0]*q[0],q[2]*q[3]-q[0]*q[1]],
                [q[3]*q[1]-q[0]*q[2],q[3]*q[2]+q[0]*q[1],q[3]*q[3]+q[0]*q[0]]]))

# Quaternions:
def q_from_T(T):
    return q_from_R(R_from_T(T))

def q_from_R(R):
    A = [1.0 + R[0][0] + R[1][1] + R[2][2],
         1.0 + R[0][0] - R[1][1] - R[2][2],
         1.0 - R[0][0] + R[1][1] - R[2][2],
         1.0 - R[0][0] - R[1][1] + R[2][2]]
    i = A.index(max(A))
    A = A[i]
    c = 0.5/np.sqrt(A)
    if   (i == 0):
        q = c*np.array([A, R[2][1]-R[1][2], R[0][2]-R[2][0], R[1][0]-R[0][1]])
    elif (i == 1):
        q = c*np.array([R[2][1]-R[1][2], A, R[1][0]+R[0][1], R[0][2]+R[2][0]])
    elif (i == 2):
        q = c*np.array([R[0][2]-R[2][0], R[1][0]+R[0][1], A, R[2][1]+R[1][2]])
    else:
        q = c*np.array([R[1][0]-R[0][1], R[0][2]+R[2][0], R[2][1]+R[1][2], A])
    return q

# Transform Matrices:
def T_from_Rp(R, p):
    return np.vstack((np.hstack((R,p)),
                      np.array([0.0, 0.0, 0.0, 1.0])))

def T_from_origin(origin):
    return T_from_Rp(R_from_rpy(origin.rpy), p_from_xyz(origin.xyz))
def T_from_axisangle(axis, theta):
    return T_from_Rp(R_from_axisangle(axis, theta), np.zeros((3,1)))	
    
def theta3DOF_from_x(point):
	# For a three DOF arm, just use derivation from HW 2 P3c
	if (len(point) != 3):
		rospy.logerr("3 DOF Point was not inputted.")
		return
	(x,y,z) = point
	r = math.sqrt(x**2+y**2)
	theta_pan = math.atan2(-x/r,y/r)
	theta_2 = math.acos((r**2+z**2+2)/2)
	theta_1 = math.atan2(z,r)-math.atan2(math.sin(theta_2),1+math.cos(theta_2))
	
	return (theta_pan, theta_1, theta_2)
#
#
#   Kinematics Class
#
#   This encapsulates the kinematics functionality, storing the
#   kinematic chain elements.
#
class Kinematics:
    def __init__(self, robot, baseframe, tipframe):
        # Report.
        rospy.loginfo("Kinematics: Setting up the chain from '%s' to '%s'...",
                      baseframe, tipframe)

        # Create the list of joints from the base frame to the tip
        # frame.  Search backwards, as this could be a tree structure.
        self.joints = []
        frame = tipframe
        while (frame != baseframe):
            joint = next((j for j in robot.joints if j.child == frame), None)
            if (joint is None):
                rospy.logerr("Unable find joint connecting to '%s'", frame)
                raise Exception()
            if (joint.parent == frame):
                rospy.logerr("Joint '%s' connects '%s' to itself",
                             joint.name, frame)
                raise Exception()
            self.joints.insert(0, joint)
            frame = joint.parent

        # Report.
        self.dofs = sum(1 for j in self.joints if j.type != 'fixed')
        rospy.loginfo("Kinematics: %d active DOFs, %d total steps",
                      self.dofs, len(self.joints))
 
    def ikin(self, x_goal, theta_0):
        if (len(theta_0) != self.dofs):
            rospy.logerr("Number of given joint angles (%d) does not match URDF (%d)",
                            len(theta_0), self.dofs)
            return
			
		# Initialize important variables
        count = 0
        delta_x = 1
        epsilon = 10**-6
        theta_ikin = theta_0
        
        while (np.linalg.norm(delta_x) > epsilon):
        	if (count >= 1000):
        	    print("Did not converge")
        	    break
        	(T,J) = self.fkin(theta_ikin)
        	if (self.dofs == 3):
        	    J_inv = np.linalg.inv(J[0:3,0:3])
        	else:
        	    J_inv = np.linalg.pinv(J[0:3,:])
        	delta_x = x_goal - p_from_T(T)
        	theta_ikin = theta_ikin + (J_inv @ delta_x)
        	count += 1
        #print('theta_ikin: \n', theta_ikin)
        return theta_ikin
	
    def fkin(self, theta):
        # Check the number of joints
        if (len(theta) != self.dofs):
            rospy.logerr("Number of joint angles (%d) does not match URDF (%d)",
                         len(theta), self.dofs)
            return

        # Initialize the T matrix to walk up the chain, the index to
        # count the moving joints, and axis of rotations (in world frame)
        T     = np.eye(4)
        index = 0        	
        e_vec = np.zeros((3,self.dofs))
        joint_pos = np.zeros((3,self.dofs))
		
        # The transform information is stored in the robot's joint entries.
        for joint in self.joints:
            if (joint.type == 'fixed'):
                T = T @ T_from_origin(joint.origin)
            elif (joint.type == 'continuous'):
                T = T @ T_from_origin(joint.origin)
                T = T @ T_from_axisangle(joint.axis, theta[index])
                R_world = R_from_T(T)
                e_vec[0:3,index] = (R_world @ np.array(joint.axis).reshape((3,1))).reshape(3)
                joint_pos[0:3,index] = p_from_T(T).reshape(3)
                index += 1
            else:
                rospy.logwarn("Unknown Joint Type: %s", joint.type)


        # COMPUTING A CROSS PRODUCT.  TO MAINTAIN THE COLUMN VECTOR,
        # SPECIFY "axis=0"
        # np.cross(VECTOR1, VECTOR2, axis=0)

        # Compute the Jacobian
        J = np.zeros((6,index))
        tip_pos = p_from_T(T)
        for i in range(index):
        	J[0:3,i] = np.cross(e_vec[:,i], tip_pos.reshape(3)-joint_pos[:,i])
        	J[3:6,i] = e_vec[:,i]

        # Return the Ttip and Jacobian.
        return (T,J)


#
#  Main Code - Test (if run independently)
#
if __name__ == "__main__":
    # Prepare/initialize this node.
    rospy.init_node('kinematics')

    # Grab the robot's URDF from the parameter server.
    robot = Robot.from_parameter_server()

    # Instantiate the Kinematics
    kin = Kinematics(robot, 'world', 'tip')

    # Pick the test angles of the robot.
    theta = [np.pi/4, np.pi/6, np.pi/3]

    # Compute the kinematics.
    (T,J) = kin.fkin(theta)

    # Report.
    np.set_printoptions(precision=6, suppress=True)
    print("T:\n", T)
    print("J:\n", J)
    print("p/q: ", p_from_T(T).T, q_from_R(R_from_T(T)))
