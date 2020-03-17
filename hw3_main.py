import numpy as np
import random
import math
from utils import *

I_h = np.identity(4) # used in hat Rodrigues Formula
I_a = np.identity(6) # used in adjoint Rodrigues Formula
W = 10*np.identity(6) # noise in prediction of U

Z = np.array([[0], [0], [0]]) # 3x1 zero matrix
Zero = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]) # 3x3 zero matrix
P = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]) # projection matrix
unobser = np.array([[-1],[-1],[-1],[-1]]) # represents unobserved feature
oTr = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

# prediction of U
def prediction(v, w, tau, Ut, V_Tt):
	theta = np.linalg.norm(-tau*w)
	hat_u = np.concatenate((np.concatenate((hat(w), Z.T), axis=0), np.concatenate((v, np.array([[0]])), axis=0)), axis=1)
	pu = np.concatenate((np.concatenate((hat(w), Zero), axis=0), np.concatenate((hat(v), hat(w)), axis=0)), axis=1)
	U = exp_h(-tau*hat_u, theta).dot(Ut)
	V_U = np.dot(exp_ad(-tau*pu, theta), np.dot(V_Tt, exp_ad(-tau*pu, theta).T)) + W
	return U, V_U

# get hat of u
def hat(u):
	hatu = np.array([[0, -u[2, 0], u[1, 0]], [u[2, 0], 0, -u[0, 0]], [-u[1, 0], u[0, 0], 0]])
	return hatu

# exponential Rodrigues Formula for hat:
def exp_h(u, theta):
	if theta == 0:
		return I_h
	eu = I_h + u + (1-math.cos(theta))/(math.pow(theta, 2))*u.dot(u) + (theta-math.sin(theta))/(math.pow(theta, 3))*u.dot(u.dot(u))
	return eu

# exponential Rodrigues Formula for adjoint
def exp_ad(u, theta):
	if theta == 0:
		return I_a
	eu = I_a + (3*math.sin(theta)-theta*math.cos(theta))/(2*theta)*u + (4 - theta*math.sin(theta)-4*math.cos(theta))/(2*math.pow(theta, 2))\
		 *np.dot(u, u)+(math.sin(theta) - theta * math.cos(theta))/(2 * math.pow(theta, 3))*np.dot(u, np.dot(u, u))+\
		 (2 - theta*math.sin(theta)-2*math.cos(theta))/(2*math.pow(theta, 4))*np.dot(u, np.dot(u, np.dot(u, u)))
	return eu

# this function transforms coordinate from pixel to world frame
def pix_tran_wrf(M, zi, cam_T_imu, Ut):
	ul = zi[0:1, :]
	vl = zi[1:2, :]
	d = ul-zi[2:3, :]
	i = np.ones((1, ul.shape[1]))
	z = i*(-M[2, 3])/d
	x = (ul*z - M[0, 2]*z)/M[0, 0]
	y = (vl*z - M[1, 2]*z)/M[1, 1]
	fe = np.concatenate((np.concatenate((x, y), axis=0), z), axis=0)
	oTw = cam_T_imu.dot(Ut)
	p = np.linalg.inv(oTw).dot(oTr)[0:3, 3:4]
	m = np.linalg.inv(oTw[0:3, 0:3]).dot(fe) + p
	return m

# if landmark is not be observed before, this function can initialize it by transforming feature's pixel to wrf
def init_m(M, features_0, cam_T_imu, Ut):
	u_m = np.zeros((3, 1))
	length = features_0.shape[1]
	m = np.zeros((3, 1))
	for i in range(length):
		if (features_0[:, i:i+1] == unobser).all():
			m = np.concatenate((m, u_m), axis = 1)
		else:
			m = np.concatenate((m, pix_tran_wrf(M, features_0[:, i:i+1], cam_T_imu, Ut)), axis=1)
	return m[:, 1:length+1]

# return perturbation and jacobian with respect to previous mean of landmarks
def get_jac_and_perturbation(features_t, m_t, M, cam_T_imu, Ut): #m_t is 3Mx1, features is 4xM, Ut is 4x4
	length = features_t.shape[1]
	m_t = m_t.reshape(length, 3).T # m_t is 3xM
	z = np.zeros((4, 1))
	a = [] # 1xNt index as j
	for i in range(length):
		fi = features_t[:, i:i + 1]
		m_j = m_t[:, i:i + 1]
		if (fi != unobser).any():
			z = np.concatenate((z, fi), axis=1)
			a.append(i)
			if (m_j == Z).all():
				m_t[:, i:i+1] = init_m(M, fi, cam_T_imu, Ut)
		else:
			continue
	z = z[:, 1:] # 4xNt
	Nt = z.shape[1]
	z_ = M.dot(pr_func(cam_T_imu.dot(Ut.dot(Homo(m_t[:, a]))))) # 4xNt
	H = np.zeros((4*Nt, 3*length))
	V = 0.001*np.eye(4*Nt)
	for i in range(Nt):
		j = a[i]
		H[4*i:4*(i+1), j:j+3] = M.dot(pr_der(cam_T_imu.dot((Ut.dot(Homo(m_t[:, j:j+1])))))).dot(cam_T_imu).dot(Ut).dot(P.T)
	return z, z_, H, V

# calculate pi()
def pr_func(q):
	return q/q[2:3, :]

# calculate derivative of pi()
def pr_der(q):
	re = np.eye(4)
	re[2, 2] = 0
	re[0, 2] = -q[0]/q[2]
	re[1, 2] = -q[1]/q[2]
	re[3, 2] = -q[3]/q[2]
	return re/q[2]

# get Homogeneous coordinates:
def Homo(q):
	length = q.shape[1]
	ii = np.ones((1,length))
	q = np.concatenate((q,ii),axis=0)
	return q

if __name__ == '__main__':
	filename = "./data/0034.npz"
	t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(filename)
	M = np.array([[K[0, 0], 0, K[0, 2], 0],
				  [0, K[1, 1], K[1, 2], 0],
				  [K[0, 0], 0, K[0, 2], -K[0, 0]*b],
				  [0, K[1, 1], K[1, 2], 0]])
	length = t.shape[1]

# This is part(a):
	#initialize U and Var of U
	Ut = np.identity(4)
	U = Ut
	U_Vt = np.zeros((6, 6))
	U_V = U_Vt

	# This is prediction only----return U and Var of U, they are 4x4xLength and 6x6xLength
	for i in range(length-1):
		delta_t = t[0, i+1] - t[0, i]
		Ut, U_Vt = prediction(linear_velocity[:, i:i+1], rotational_velocity[:, i:i+1], delta_t, Ut, U_Vt)
		U = np.dstack((U, Ut))
		#U_V = np.dstack((U_V, U_Vt))
	np.save("pred0034.npy", U)

	# interval
	interval = 5
	f_t = features[:, ::interval, 0]

	M_num = f_t.shape[1] # number of samples

# This is part(b)
	#initalize mt and var of mt--mt and mvt is respectly mean and variance of landmark pose at time t
	m_0 = init_m(M, f_t, cam_T_imu, U[:, :, 0])
	m_0 = m_0.flatten('F')
	m_0 = m_0.reshape(3*M_num,1)
	m_V_0 = 0.01*np.eye(3*M_num)
	m_h = m_0  # m_h is used to store m of every timestamp-----it is useless and slows my code
	m_V_h = m_V_0 # m_V_h is used to store var of m of every timestamp-----it is useless and slows my code
	m_t = m_0
	m_V_t = m_V_0

	#This is update
	for i in range(1, length):
		if i % 50 == 0:
			print('this is ', i / 50, 'times loop')
		f_t = features[:, ::interval, i]
		z_t, z_, H, V = get_jac_and_perturbation(f_t, m_t, M, cam_T_imu, U[:,:,i])
		z_t = z_t.flatten('F')
		z_t = z_t.reshape(z_t.shape[0], 1)
		z_ = z_.flatten('F')
		z_ = z_.reshape(z_.shape[0], 1)
		K = m_V_t.dot(H.T).dot(np.linalg.inv(H.dot(m_V_t).dot(H.T)+V))
		m_t = m_t + K.dot(z_t-z_)
		#m_V_t = (np.eye(3*M_num) - K.dot(H)).dot(m_V_t)      it is useless and slow my code
		#m_h = np.concatenate((m_h,m_t), axis=1)         it is useless and slows my code
		#m_V_h = np.dstack((m_V_h, m_V_t))      it is useless and slows my code
	np.save("update0034.npy", m_t)

