#%% Imports

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
from scipy.linalg import expm

def nominal_state_update(nominal_state, w_m, a_m, dt):
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state
    R = q.as_matrix()

    new_p = p + v*dt + 0.5*(R@(a_m-a_b)+g)*dt**2
    new_v = v + (R@(a_m-a_b)+g)*dt
    new_q = q*Rotation.from_rotvec(((w_m-w_b)*dt).reshape((3,)))

    return new_p, new_v, new_q, a_b, w_b, g


def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state
    R = q.as_matrix()

    V_i = accelerometer_noise_density**2*dt**2*np.identity(3)
    Th_i = gyroscope_noise_density**2*dt**2*np.identity(3)
    A_i = accelerometer_random_walk**2*dt*np.identity(3)
    Om_i = gyroscope_random_walk**2*dt*np.identity(3)

    Q_i1 = np.hstack((V_i,np.zeros((3,9))))
    Q_i2 = np.hstack((np.zeros((3,3)),Th_i,np.zeros((3,6))))
    Q_i3 = np.hstack((np.zeros((3,6)),A_i,np.zeros((3,3))))
    Q_i4 = np.hstack((np.zeros((3,9)),Om_i))
    Q_i = np.vstack(((Q_i1,Q_i2,Q_i3,Q_i4)))

    F_i1 = np.zeros((3,12))
    F_i2 = np.hstack((np.identity(3),np.zeros((3,9))))
    F_i3 = np.hstack((np.zeros((3,3)),np.identity(3),np.zeros((3,6))))
    F_i4 = np.hstack((np.zeros((3,6)),np.identity(3),np.zeros((3,3))))
    F_i5 = np.hstack((np.zeros((3,9)),np.identity(3)))
    F_i6 = np.zeros((3,12))
    F_i = np.vstack((F_i1,F_i2,F_i3,F_i4,F_i5,F_i6))

    a_mb = a_m.reshape((3,)) - a_b.reshape((3,))
    skew_a_mb = np.array([[0,-a_mb[2],a_mb[1]],[a_mb[2],0,-a_mb[0]],[-a_mb[1],a_mb[0],0]])

    F_x1 = np.hstack((np.identity(3),np.identity(3)*dt,np.zeros((3,12))))
    F_x2 = np.hstack((np.zeros((3,3)),np.identity(3), -R@skew_a_mb*dt,-R*dt,np.zeros((3,3)),np.identity(3)*dt))
    F_x3 = np.hstack((np.zeros((3,6)),Rotation.from_rotvec(((w_m-w_b)*dt).reshape((3,))).as_matrix().T,np.zeros((3,3)),-np.identity(3)*dt,np.zeros((3,3))))
    F_x4 = np.hstack((np.zeros((3,9)),np.identity(3),np.zeros((3,6))))
    F_x5 = np.hstack((np.zeros((3,12)),np.identity(3),np.zeros((3,3))))
    F_x6 = np.hstack((np.zeros((3,15)),np.identity(3)))
    F_x = np.vstack((F_x1,F_x2,F_x3,F_x4,F_x5,F_x6))

    new_covariance = F_x@error_state_covariance@F_x.T + F_i@Q_i@F_i.T

    # return an 18x18 covariance matrix
    return new_covariance


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state
    R = q.as_matrix()

    # compute the innovation next state, next error_state covariance
    Pc = R.T @ (Pw-p)
    #innovation = uv-np.array([[1,0,0],[0,1,0]])@Pc
    innovation = np.array([[uv[0][0]-Pc[0][0]/Pc[2][0]],[uv[1][0]-Pc[1][0]/Pc[2][0]]])

    if np.linalg.norm(innovation) <= error_threshold:
        #dz_dPc = np.array([[1,0,-uv[0][0]],[0,1,-uv[1][0]]])/Pc[2][0]
        dz_dPc = np.array([[1,0,-Pc[0][0]/Pc[2][0]],[0,1,-Pc[1][0]/Pc[2][0]]])/Pc[2][0]
        dPc_dth = np.array([[0, -Pc[2][0], Pc[1][0]], [Pc[2][0], 0, -Pc[0][0]], [-Pc[1][0], Pc[0][0], 0]])
        dPc_dp = -R.T
        dz_dp = dz_dPc@dPc_dp
        dz_dth = dz_dPc@dPc_dth
        H_t = np.hstack((dz_dp,np.zeros((2,3)),dz_dth,np.zeros((2,9))))
        K_t = error_state_covariance@H_t.T@np.linalg.inv(H_t@error_state_covariance@H_t.T+Q)
        dx = K_t@innovation
        error_state_covariance = (np.identity(18)-K_t@H_t)@error_state_covariance@(np.identity(18)-K_t@H_t).T + K_t@Q@K_t.T

        p += dx[0:3,:]
        v += dx[3:6,:]
        q *= Rotation.from_rotvec(dx[6:9,:].reshape((3,)))
        a_b += dx[9:12,:]
        w_b += dx[12:15,:]
        g += dx[15:18,:]

    return (p, v, q, a_b, w_b, g), error_state_covariance, innovation
