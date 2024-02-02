from time import *
import numpy as np
import pandas as pd
import math
import copy
#import vector

#from csv import reader
from statistics import *
from decimal import Decimal
from fractions import Fraction
#from scipy.linalg import norm
from pandas import DataFrame
from pyscript import document


def noiseModel():

    ARW = [0.0112 * pi / 180, 0.0047 * pi / 180, 0.0039 * pi / 180] 
    ARRW = [9.53e-5 * pi / 180, 1.85e-4 * pi / 180, 8.30e-5 * pi / 180]
    RW = [0.040/1000, 0.031/1000, 0.040/1000] # convert to g's
    RRW = [6.15e-4/1000, 0.002/1000, 0.002/1000] #convert to g's


    
    M_cov_gyros_dt = [ARW[0]**2, ARW[1]**2, ARW[2]**2]
    M_cov_gyro_bias_dt = [ARRW[0]**2, ARRW[1]**2, ARRW[2]**2]
    M_cov_acc_dt = [RW[0]**2, RW[1]**2, RW[2]**2]
    M_cov_acc_bias_dt = [RRW[0]**2, RRW[1]**2, RRW[2]**2]
    
    return M_cov_gyros_dt, M_cov_gyro_bias_dt, M_cov_acc_dt, M_cov_acc_bias_dt

def attitude_from_acc(IMU_accx, IMU_accy, IMU_accz):
    phi   = float(math.atan2( IMU_accy, IMU_accz))
    theta = float(math.atan2(-IMU_accx, math.sqrt(IMU_accy**2 + IMU_accz**2)))
    return phi, theta #roll and pitch orientation respectively...

def calculateM_dt(phi, theta, dt):
    M_dt = [ [ dt*1.0, dt*math.sin(phi)*math.tan(theta), dt*math.cos(phi)*math.tan(theta)],
            [ dt*0.0,            dt*math.cos(phi),           dt * -math.sin(phi)] ]
    M_dt = np.array(M_dt)
    
    dM_dphi = [ [ dt*0, dt*math.cos(phi)*math.tan(theta), dt*-math.sin(phi)*math.tan(theta)],
               [ dt*0,           dt*-math.sin(phi),            dt*-math.cos(phi)] ]
    dM_dphi = np.array(dM_dphi)
    
    dM_dtheta = [ [ dt*0, dt*math.sin(phi)*(math.tan(theta)**2 + 1), dt*math.cos(phi)*(math.tan(theta)**2 + 1)],
                 [ dt*0,                           dt*0,                           dt*0] ]
    dM_dtheta = np.array(dM_dtheta)
    
    return M_dt, dM_dphi, dM_dtheta

def calculateC_dt(phi, theta, dt):
    C_dt = [ [  dt*math.cos(theta), dt*math.sin(phi)*math.sin(theta), dt*math.cos(phi)*math.sin(theta)],
            [           dt*0,            dt*math.cos(phi),           dt*-math.sin(phi)],
            [ dt*-math.sin(theta), dt*math.cos(theta)*math.sin(phi), dt*math.cos(phi)*math.cos(theta)] ]
    C_dt = np.array(C_dt)

    dC_dphi = [ [ dt*0, dt*math.cos(phi)*math.sin(theta), dt*-math.sin(phi)*math.sin(theta)],
                   [ dt*0,           dt*-math.sin(phi),            dt*-math.cos(phi)], 
                   [ dt*0, dt*math.cos(phi)*math.cos(theta), dt*-math.cos(theta)*math.sin(phi)] ]
    dC_dphi = np.array(dC_dphi)

    dC_dtheta = [ [ dt*-math.sin(theta),  dt*math.cos(theta)*math.sin(phi),  dt*math.cos(phi)*math.cos(theta)],
                     [           dt*0,                    dt*0,                    dt*0],
                     [ dt*-math.cos(theta), dt*-math.sin(phi)*math.sin(theta), dt*-math.cos(phi)*math.sin(theta)] ]
    dC_dtheta = np.array(dC_dtheta)
    
    return C_dt, dC_dphi, dC_dtheta

po_H = np.zeros((3,8))
po_H[0:3,5:8] = np.identity(3, dtype="float")
po_V = np.identity(3, dtype="float")
po_a = np.array([0, 0, 1]) 
po_a = po_a.reshape(-1,1)
def processOutput(x, u_accel, dt):
    global po_H
    global po_V
    global po_a
    
    #% attitude
    # phi = x(1); theta = x(2);
    phi = x[0,0]
    theta = x[1,0]
    '''   
    % x(2:5): gyro bias
    % x(5:8): low-pass accelerations

    % u_accel(0:3): accel readings in g's

    % Zero-constraint equation as output:
    % linear low-pass accelerations should be zero
    % (i.e., low-pass total acceleration should equal [0 0 1]')
    '''
    y = np.array([x[5,0], x[6,0], x[7,0]])
    y =  y.reshape(-1,1)

#     % Jacobian of the output equation with respect to the state, 
#     % evaluated at current (x, u, w = 0)             
#     H = [[column for column in range(8)] for row in range(3)]n--
#     H = np.array(H) * 0n--

#     H = np.zeros((3,8))
#     H[0:3,5:8] = np.identity(3, dtype="float")

#     % Jacobian of the output equation with respect to noise affecting 
#     % the output, evaluated at current (x, u, w = 0)             
#     V = np.identity(3, dtype="float")

#     % high-pass linear accelerations
    C, dC_dphi, dC_dtheta = calculateC_dt(phi, theta, 1)
#     y_hp = C * u_accel - a - y
    y_hp = np.dot(C, u_accel) - po_a - y
    
#     y = np.matrix(y)
#     y_hp = np.matrix(y_hp)
    
    return y, po_H, po_V, y_hp

st_ag = np.identity(2, dtype="float")
st_eye = np.identity(3, dtype="float")
st_b = np.array([0,0,1])
st_b = st_b.reshape(-1,1)
def stateTransition(x, u, dt, Ka):
    global st_ag
    global st_eye
    global st_b
    
    phi = x[0,0]
    theta = x[1,0]
    #print(u)
    '''
    Data structure...
    % x(2:5): gyro bias
    % x(5:8): low-pass accelerations
    % % % % x(8:11): accel bias

    % u(0:3): gyro readings in rad/s
    % u(3:6): accel readings in g's
    '''
    
    #gyro bias are given by states x[2,0], x[3,0], x[4,0] 
    #% bias corrected gyro measurements
    p = u[0,0] - x[2,0]
    q = u[1,0] - x[3,0]
    r = u[2,0] - x[4,0]
    pqr = np.array([p,q,r])
    pqr = pqr.reshape(-1,1)
    
    #bias corrected accel measurements...
    #u_acc = [u[3,0] - x[11,0], u[4,0] - x[12,0], u[5,0] - x[13,0]]
    u_acc = [u[3,0], u[4,0], u[5,0]]
    u_acc = np.array(u_acc)
    u_acc = u_acc.reshape(-1,1)
    
    #MatLab code:
    #[M_dt,dM_dphi_dt, dM_dtheta_dt] = calculateM_dt(phi, theta, dt);
    M_dt, dM_dphi_dt, dM_dtheta_dt = calculateM_dt(phi, theta, dt)
#     delta_att = M_dt * pqr
    delta_att = np.dot(M_dt, pqr)

    #MatLab code:
#     % these are not scaled by dt
#     [C, dC_dphi, dC_dtheta] = calculateC_dt(phi, theta, 1);
    C, dC_dphi, dC_dtheta = calculateC_dt(phi, theta, 1)
    
    '''
    % Jacobian of process equation with respect to the state, 
    % evaluated at current (x, u, w = 0):           
    A = zeros(8,8);
    A(1:2,1:2) = eye(2) + [dM_dphi_dt*pqr dM_dtheta_dt*pqr];
    A(1:2,3:5) = -M_dt;
    A(3:5,3:5) = eye(3);
    A(6:8,1:2) = (eye(3) - Ka) * [dC_dphi*u_acc dC_dtheta*u_acc];
    A(6:8,6:8) = Ka;
    '''
    A = np.identity(8, dtype="float") * 0.0
    A[0:2,0] = st_ag[:,0] + np.dot(dM_dphi_dt,pqr[:,0])
    A[0:2,1] = st_ag[:,1] + np.dot(dM_dtheta_dt,pqr[:,0])
    A[0:2,2:5] = -M_dt
    A[2:5,2:5] = st_eye

    dC_dphi_u_acc = np.dot(dC_dphi,u_acc[:,0])
    dC_dtheta_u_acc = np.dot(dC_dtheta,u_acc[:,0])
#     A[5:8,0:2] = (eye - Ka) @ np.concatenate((dC_dphi_u_acc, dC_dtheta_u_acc), axis=1)
    A[5:8,0:2] = np.dot((st_eye - Ka),  np.vstack((dC_dphi_u_acc, dC_dtheta_u_acc)).T)#, axis=1)
    A[5:8,5:8] = Ka
    
    '''
    % Jacobian of process equation with respect to the noise affecting
    % the process, evaluated at current (x, u, w = 0)             
    W = zeros(8,9);
    W(1:2, 1:3) =        M_dt;
    W(3:5,4:6) =    dt * eye(3);
    W(6:8,7:9) = (eye(3) - Ka) * C;
    '''
    W = np.zeros((8,9))
    W[0:2,0:3] = M_dt
    W[2:5,3:6] = dt*st_eye
    W[5:8,6:9] = (st_eye - Ka) @ C
    #pd.DataFrame(W)
    
    '''
    % Next state
    % (remove gravity in global coordinates)
    acc = C * u_acc  - [0 0 1]';  

    x = [ [phi; theta] + delta_att;
       x(3:5); % gyro_bias(k+1) = gyro_bias(k)
       Ka * x(6:8) + (eye(3) - Ka) * acc ];
    '''
    acc = np.dot(C, u_acc)  - st_b
    
    phi = phi + delta_att[0]
    theta = theta + delta_att[1]
#     acc_bias = np.dot(Ka, x[5:8]) + np.dot((eye - Ka), acc)
    acc_bias = Ka @ x[5:8] + (eye - Ka) @ acc
#     acc_bias = Ka * x[5:8] + (eye - Ka) * acc
    x = np.array([phi[0], theta[0],\
         x[2,0], x[3,0], x[4,0],\
         acc_bias[0,0], acc_bias[1,0], acc_bias[2,0]])
    x = x.reshape(-1,1)

#     x = np.matrix(x)
#     A = np.matrix(A)
#     W = np.matrix(W)
    
    return x, A, W


dt = IMU_dt

def loadOtherData(event):
    with open('shared_data.txt','r') as file:
        exec(file.read())
        # print("IMU_acc_x:", IMU_acc_x)
        output_ta = document.querySelector("#output2")
        output_ta.innerText = str(IMU_acc_x)





# print("IMU_acc_x:", IMU_acc_x)
# print("IMU_acc_y:", IMU_acc_y)
# print("IMU_acc_z:", IMU_acc_z)
# print("IMU_gyro_x:", IMU_gyro_x)
# print("IMU_gyro_y:", IMU_gyro_y)
# print("IMU_gyro_z:", IMU_gyro_z)
# print("IMU_dt11:", IMU_dt)    

#% this indirectly controls the smoothing of the attitude estimate
Ka = 0.9956 * np.identity(3, dtype="float") #% 2.0 s time constant

# % zero-condition weight 
# %(this more or less controls the tightness of the gyro drift estimation) 
ZCW = 2000.0

pi = 3.1415926535897932384626433832795

M_cov_gyros_dt, M_cov_gyro_bias_dt, M_cov_acc_dt, M_cov_acc_bias_dt = noiseModel()

# % What matters is how strong measurement noise is with respect to process
# % noise. We scale both R and Q by a common scaling factor to make the 
# % numerical calculation of P more stable:
rescaling = 1000.0

# % Output noise covariance matrix:
# % (this controls how much weight to give to the zero-condition constraint )
# % ( the process output y is in g's, so the factor of 1000 makes sense )
eye = np.identity(3, dtype="float")
# R = 1000.0 * rescaling * eye * (np.array(M_cov_acc_dt) / dt )
R = ZCW * rescaling * eye * (np.array(M_cov_acc_dt) / dt )

# % Process noise covariance matrix:
eye_q = np.identity(9, dtype="float")
Q = rescaling * eye_q * (np.array([M_cov_gyros_dt, M_cov_gyro_bias_dt, M_cov_acc_dt ]).flatten() / dt)

phi_acc, theta_acc = attitude_from_acc(IMU_acc_x[0], IMU_acc_y[0], IMU_acc_z[0])
phi_init = phi_acc
theta_init = theta_acc
#display(md(f'<hr style="border:0.5px solid black">'))
#display(md(f'<span style="font-family:Ariel;font-size:20px">**Initial $\Phi$ :** {np.round(phi_init * 180/pi,2)} deg, **Initial $\Theta$:** {np.round(theta_init * 180/pi,2)} deg</span>'))

#% everything
NUM_IMU_READINGS = len(IMU_t)
#display(md(f'<span style="font-family:Ariel;font-size:20px">**NUM_IMU_READINGS:** {NUM_IMU_READINGS}</span>'))

# % initialization
x = np.array([phi_init, theta_init, 0, 0, 0, 0, 0, 0])
x = x.reshape(-1,1)

# %Initialize P to diagonal matrix of smaller magnitude than Q
# accomodate the initialization with given Po (P = Po;) once values available using widget input
P = np.identity(len(x), dtype=np.float64) * np.linalg.norm(Q, 2)
# P = np.asmatrix(P)
# pd.DataFrame(P)

I_mat = np.identity(len(P), dtype=np.float64)
# I_mat = np.asmatrix(I_mat)

p_acc = np.array([IMU_acc_x[0], IMU_acc_y[0], IMU_acc_z[0]])/1000
p_acc = p_acc.reshape(-1,1)
y, H, V, y_hp = processOutput(x, p_acc, dt)
# print(y, H, V, '\ny_hp:\n',y_hp)

#we will store the augmented state estimates and other adjacent attributes like CORRECTION and off-course OUTPUT...
STATE = []#keep a record of current state, will keep it in a list and keep appending...
OUTPUT = []
CORRECTION = []

# % store the state and output (add y_hp to the state)
# temp = [*x.flatten().tolist()[0], *y_hp.flatten().tolist()[0]]
# temp = np.concatenate((x,y_hp),axis=0)
temp =[*(x.T).flatten(), *(y_hp.T).flatten()]
STATE.append(temp)
# print(STATE)
# temp = [*y.flatten().tolist()[0], *y_hp.flatten().tolist()[0]]
# temp = np.concatenate((y,y_hp),axis=0)
temp =[*(y.T).flatten(), *(y_hp.T).flatten()]
OUTPUT.append(temp)

k = 0 #Python starts with index 0 unlike MatLab... 
while True:    
#     % Kalman filter to compute state estimate for k+1
    
    if (k >= NUM_IMU_READINGS-1):
        break
    
#     % % % prediction step
    
#     % get gyro readings (deg/s) ==> convert to rad/s
    p_acc = np.array([IMU_acc_x[k], IMU_acc_y[k], IMU_acc_z[k]]) / 1000
    p_gyro = np.array([IMU_gyro_x[k], IMU_gyro_y[k], IMU_gyro_z[k]]) * pi/180
    u = np.concatenate((p_gyro,p_acc),axis=0)
    u = u.reshape(-1,1)
    
    x, A, W = stateTransition(x,u,dt,Ka)
#     % (x is now the prediction for k+1) 
    
#     % Project error covariance
#     P = A * P * A' + W * Q * W';
#     P = A * P * np.matrix.transpose(A) + W * Q * np.matrix.transpose(W)
    P = A @ P @ A.T + W @ Q @ W.T
       
    
#     % % %  measurement update for k+1:
    
#     % get accelerometer readings and convert to g's
#     u = [IMU_acc_x(k+1); IMU_acc_y(k+1); IMU_acc_z(k+1)] / 1000
    p_acc = np.array([IMU_acc_x[k+1], IMU_acc_y[k+1], IMU_acc_z[k+1]])/1000
    p_acc = p_acc.reshape(-1,1)
        
    y, H, V, y_hp = processOutput(x, p_acc, dt)
    
#     % Kalman gain:
#     S = H * P * H' + V * R * V';
#     K = P * H' / S;
    S = H @ P @ H.T + V @ R @ V.T
    #Code below translates to K = P * H'/ S;
    St = np.linalg.inv(S)
    K = (P @ H.T)
    K = K @ St 

    #     % output is supposed to be zeros
    x = x + K @ (0 - y)
    P = (I_mat - K @ H) @ P

    #     % store correction
    #     CORRECTION(k,:) = (-K*y)'; 
    CORRECTION.append((-K@y).flatten().tolist())
    
#     % compute the updated output for *storage*
    y, H, V, y_hp = processOutput(x, p_acc, dt)
    
#     % store as next value k + 1
    k = k + 1
#     % store estimates (add y_hp to the state)
#     STATE(k,:) = [x; y_hp]';
#     OUTPUT(k,:) = [y; y_hp]';
#     temp = np.concatenate((x,y_hp),axis=0) #[*x.flatten().tolist()[0], *y_hp.flatten().tolist()[0]]
    temp =[*(x.T).flatten(), *(y_hp.T).flatten()]
    STATE.append(temp)
    #print('temp1:', temp)
#     temp = np.concatenate((y,y_hp),axis=0) #[*y.flatten().tolist()[0], *y_hp.flatten().tolist()[0]]
    temp = [*(y.T).flatten(), *(y_hp.T).flatten()]
    OUTPUT.append(temp)
    
STATE = np.matrix(STATE)
# pd.DataFrame(STATE)
STATE[:,0:5] = STATE[:,0:5] * 180/pi

# % Low-pass and high-pass linear accelerations
STATE[:,5:11] = STATE[:,5:11] * 1000 #to produce speed in mg

STATE = np.matrix(STATE)

# % convert to mg the acceleration output
OUTPUT = np.matrix(OUTPUT) * 1000
# % Note: The high-pass portion of acceleration is useful to 
# % count walking steps
CORRECTION = np.matrix(CORRECTION)

# print(STATE) uncomment later