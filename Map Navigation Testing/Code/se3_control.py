import numpy as np
from scipy.spatial.transform import Rotation

class SE3Control(object):
    def __init__(self, quad_params):
        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s

        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2
        self.gamma = self.k_drag/self.k_thrust
        # Geometric Control
        self.k_p = 0.7*np.diag([8.3, 8.3, 8.3])
        self.k_d = 0.7*np.diag([4.5, 4.5, 4.5])
        self.k_R = 0.7*np.diag([218, 218, 218])
        self.k_w = 0.7*np.diag([20, 20, 21])

    def vee(self, matrix):
        return np.array([[matrix[2,1]],[matrix[0,2]],[matrix[1,0]]]).reshape([3,1])

    def update(self, t, state, flat_output):
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))
        """
        # Linear Controller
        Rot = Rotation.from_quat(state['q'])
        euler = Rot.as_euler('zxy')

        r_ddot_des = flat_output['x_ddot'] - self.k_d*(state['v'] - flat_output['x_dot']) - self.k_p*(state['x'] - flat_output['x'])
        u1 = self.mass*(r_ddot_des[2]+self.g)

        phi_des = (r_ddot_des[0]*np.sin(flat_output['yaw'] - r_ddot_des[1]*np.cos(flat_output['yaw'])))/self.g
        theta_des = (r_ddot_des[0]*np.cos(flat_output['yaw'] + r_ddot_des[1]*np.sin(flat_output['yaw'])))/self.g
        u2 = np.matmul(self.inertia, np.array([[-self.k_p_phi*(euler[0]-phi_des)-self.k_d_phi*(state['w'][0]-0)],
                                               [-self.k_p_th*(euler[1]-theta_des)-self.k_d_th*(state['w'][1]-0)],
                                               [-self.k_p_psi*(euler[2]-flat_output['yaw'])-self.k_d_psi*(state['w'][2]-flat_output['yaw_dot'])]]))

        F1 = (u1 - 2*u2[1]/self.arm_length + u2[2]/self.gamma)/4
        F2 = -F1 + u1/2 + (u2[0]-u2[1])/(2*self.arm_length)
        F3 = F1 + u2[1]/self.arm_length
        F4 = F2 - u2[0]/self.arm_length
        forces = np.array([F1, F2, F3, F4])
        motor_speeds = np.sqrt(forces/self.k_thrust)

        cmd_thrust = u1
        for i in range(len(cmd_moment)):
            cmd_moment[i] = u2[i]
        for j in range(len(cmd_motor_speeds)):
            cmd_motor_speeds[j] = motor_speeds[j]
        q = Rotation.from_euler('zxy',[phi_des,theta_des,flat_output['yaw']])
        cmd_q = q.as_quat()
        """
        # Geometric Controller
        r_ddot_des = flat_output['x_ddot'] - self.k_d@(state['v'] - flat_output['x_dot']) - self.k_p@(state['x'] - flat_output['x'])
        F_des = (self.mass*r_ddot_des + np.array([0,0,self.mass*self.g])).reshape([3,1])
        R = np.asarray(Rotation.from_quat(state['q']).as_matrix())
        b3 = R @ np.array([0,0,1]).T
        u1 = b3.T @ F_des

        b3_des = F_des/np.linalg.norm(F_des)
        a_yaw = np.array([[np.cos(flat_output['yaw'])],[np.sin(flat_output['yaw'])],[0]])
        b2_des = np.cross(b3_des,a_yaw,axis=0)/np.linalg.norm(np.cross(b3_des,a_yaw,axis=0))
        #R_des = np.array([np.cross(b2_des,b3_des,axis=0),b2_des,b3_des]).reshape([3,3])
        R_des = np.hstack([np.cross(b2_des.T,b3_des.T).T,b2_des,b3_des])

        e_R = 0.5*self.vee(R_des.T@R - R.T@R_des)
        e_w = (state['w'] - flat_output['yaw_dot']).reshape([3,1])

        u2 = np.matmul(self.inertia, (-self.k_R@e_R - self.k_w@e_w))

        R_motor_input = np.array([[self.k_thrust, self.k_thrust, self.k_thrust, self.k_thrust],
                                  [0, self.k_thrust*self.arm_length, 0, -self.k_thrust*self.arm_length],
                                  [-self.k_thrust*self.arm_length, 0, self.k_thrust*self.arm_length, 0],
                                  [self.k_drag, -self.k_drag, self.k_drag, -self.k_drag]])
        inputs = np.array([u1,u2[0],u2[1],u2[2]])

        motor_speeds_sq = np.linalg.inv(R_motor_input) @ inputs
        for i in range(len(motor_speeds_sq)):
            if motor_speeds_sq[i] < self.rotor_speed_min**2:
                motor_speeds_sq[i] = self.rotor_speed_min**2
            elif motor_speeds_sq[i] > self.rotor_speed_max**2:
                motor_speeds_sq[i] = self.rotor_speed_max**2
        motor_speeds = np.sqrt(motor_speeds_sq)
        """
        F1 = (u1 - 2 * u2[1] / self.arm_length + u2[2] / self.gamma) / 4
        F2 = -F1 + u1 / 2 + (u2[0] - u2[1]) / (2 * self.arm_length)
        F3 = F1 + u2[1] / self.arm_length
        F4 = F2 - u2[0] / self.arm_length
        forces = np.array([F1, F2, F3, F4])
        motor_speeds = np.sqrt(forces / self.k_thrust)
        """

        cmd_thrust = u1
        for i in range(len(cmd_moment)):
            cmd_moment[i] = u2[i]
        for j in range(len(cmd_motor_speeds)):
            cmd_motor_speeds[j] = motor_speeds[j]
        cmd_q = Rotation.from_matrix(R_des).as_quat()

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input
