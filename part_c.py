import hw3_main as hm
from utils import *
import numpy as np

# calculate u (6x1) in form of hat
def get_hat(u):
    return np.concatenate((np.concatenate((hm.hat(u[3:6, :]), hm.Z.T), axis=0), np.concatenate((u[0:3,:], np.array([[0]])), axis=0)), axis=1)

# calculate u circle
def assoc(u):
    s = np.zeros((4, 6))
    s[0:3, 0:3] = np.eye(3)
    s[0:3, 3:6] = -hm.hat(u[0:3, :])
    return s

# calculate perturbation and jointly updated jacobian
def get_jac_and_perturbation(features_t, m_t, M, cam_T_imu, Ut): # m_t is previous mean of landmark poses, Ut is IMU inverse pose, M is Intrinsic Calibration, cam_T_imu is Extrinsic Calibration
    length = features_t.shape[1]
    m_t = m_t.reshape(length, 3).T  # change m_t's size to 3xM
    z = np.zeros((4, 1))
    a = []  # to store index of m_t
    for i in range(length):
        fi = features_t[:, i:i + 1]
        m_j = m_t[:, i:i + 1]
        if (fi != hm.unobser).any(): # if feature is observed, then use it to update
            z = np.concatenate((z, fi), axis=1)
            a.append(i)
            if (m_j == hm.Z).all(): # if m_t(i) corresponding to this feature is not initialized before, then initialize it
                m_t[:, i:i + 1] = hm.init_m(M, fi, cam_T_imu, Ut)

    z = z[:, 1:]  # observed feature

    Nt = z.shape[1] # number of observed feature

    # predict feature based on predicted Ut and corresponding m_t
    z_ = M.dot(hm.pr_func(cam_T_imu.dot(Ut.dot(hm.Homo(m_t[:, a])))))

    # calculate jacobian H1 with respect to feature pose evaluated at m_t and jacobian H2 with respect to inverse IMU pose evaluated at Ut
    H1 = np.zeros((4 * Nt, 3 * length))
    H2 = np.zeros((4 * Nt, 6))
    for i in range(Nt):
        j = a[i]
        ref = M.dot(hm.pr_der(cam_T_imu.dot(Ut).dot(hm.Homo(m_t[:, j:j + 1])))).dot(
            cam_T_imu)
        H1[4 * i:4 * (i + 1), j:j + 3] = ref.dot(Ut).dot(hm.P.T)
        H2[4 * i:4 * (i + 1), :] = ref.dot(assoc(Ut.dot(hm.Homo(m_t[:, j:j + 1]))))
    return z, z_, H1, H2

if __name__ == '__main__':
    filename = "./data/0034.npz"
    t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(filename)
    M = np.array([[K[0, 0], 0, K[0, 2], 0],
                  [0, K[1, 1], K[1, 2], 0],
                  [K[0, 0], 0, K[0, 2], -K[0, 0] * b],
                  [0, K[1, 1], K[1, 2], 0]])
    length = t.shape[1]

    # initialize mean of U and landmarks separately. and initialize covariance of U and landmark
    interval = 5
    U_0 = np.identity(4)
    f_0 = features[:, ::interval, 0]
    M_num = f_0.shape[1]
    m_0 = hm.init_m(M, f_0, cam_T_imu, U_0)
    m_0 = m_0.flatten('F')
    m_0 = m_0.reshape(3 * M_num, 1)
    cor_0 = 0.01*np.eye(3*M_num+6)

    U_t = U_0
    m_t = m_0
    cor_t = cor_0
    U = U_t

    # IV SLAM
    for i in range(length-1):
        if i%50 == 0:
            print('this is ',i/50, 'times loop')
        f_t = features[:, ::interval, i+1]
        delta_t = t[0, i + 1] - t[0, i]

        # prediction step of IMU
        U_t_p, cor_t[3*M_num:3*M_num+6, 3*M_num:3*M_num+6] = hm.prediction(linear_velocity[:, i:i + 1], rotational_velocity[:, i:i + 1], delta_t, U_t, cor_t[3*M_num:3*M_num+6, 3*M_num:3*M_num+6])

        # update step of IMU and landmark
        z_t, z_, H1, H2 = get_jac_and_perturbation(f_t, m_t, M, cam_T_imu, U_t_p) # H1 is 4Ntx3M, H2 is 4Ntx6
        z_t = z_t.flatten('F')
        z_t = z_t.reshape(z_t.shape[0], 1) # z t+1|t is 4Ntx1
        z_ = z_.flatten('F')
        z_ = z_.reshape(z_.shape[0], 1) # z_ which is based on previous pose is 4Ntx1

        # initialize V
        V = 1000*np.eye(z_t.shape[0])

        H = np.hstack((H1, H2)) # H 4Ntx(3M+6) is jointly combined by H1 (landmark's update) and H2 (Ut's update)

        # calculate joint Kalman Gain
        inver = np.linalg.inv(H.dot(cor_t).dot(H.T) + V)
        K = cor_t.dot(H.T).dot(inver)

        # calculate joint perturbation
        perturbation = K.dot(z_t - z_)
        # divide joint perturbation into Ut's and landmark's
        perturbation_u = perturbation[3*M_num:3*M_num+6, :]
        perturbation_landmark = perturbation[0:3*M_num, :]

        hat_perturbation_u = get_hat(perturbation_u)
        theta = np.linalg.norm(perturbation_u[3:6, :])

        # update Ut
        U_t = hm.exp_h(hat_perturbation_u, theta).dot(U_t_p)

        # store Ut in U
        U = np.dstack((U, U_t))

        # update landmarks
        m_t = m_t + perturbation_landmark

        #update covariance
        cor_t = (np.eye(3*M_num+6) - K.dot(H)).dot(cor_t)
    np.save('slam0034_U', U)
    np.save('landmark0034', m_t)
