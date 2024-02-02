# *****************************************************
# Description: MatLab to Python code conversion
# Date: 9th Feb 2023
# Author: Syed Yasin
# *****************************************************

#------------------------------------------------------
# This module gets data urls from aws cloud
# importing the requests library
#import requests
import json
import csv
import numpy as np
import pandas as pd
import datetime 
from datetime import datetime
from datetime import date
from datetime import time
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from statistics import mean
from math import *
from decimal import *
import copy
import datetime
import math
import js
import time

from pyodide.http import open_url
from pyscript import document

def datetime_pytom(d):
    '''
    Description
        This function is a Python equivalent of datenum() function in MatLab
    Input
        d date in the format datetime.datetime(2022, 11, 8, 7, 38, 58, 590000)
    Output
        The fractional day count since 0-Jan-0000 (proleptic ISO calendar)
        This is the 'datenum' datatype in matlab
    Notes on day counting
        matlab: day one is 1 Jan 0000 
        python: day one is 1 Jan 0001
        hence an increase of 366 days, for year 0 AD was a leap year
    '''
    dd = date.toordinal(d) + 366
    tm = datetime.timedelta(hours=d.hour,minutes=d.minute,seconds=d.second,microseconds=d.microsecond)
    tm = datetime.timedelta.total_seconds(tm) / (24 * 3600)
    return dd + tm

def datetime_mtopy(datenum):
    '''
    Input
        The fractional day count according to datenum datatype in matlab
    Output
        The date and time as a instance of type datetime in python
    Notes on day counting
        matlab: day one is 1 Jan 0000 
        python: day one is 1 Jan 0001
        hence a reduction of 366 days, for year 0 AD was a leap year
    '''
    ii = datetime.datetime.fromordinal(int(datenum) - 366)
    ff = datetime.timedelta(days=datenum%1)
    return ii + ff

def dot_A_B_2(m1,m2):
    m1 = np.array(m1,dtype=np.double)
    m2 = np.array(m2,dtype=np.double)
    z = (m1 * m2).sum(axis=0)
#     n = z[:,0]+z[:,1]+z[:,2]
#     n = np.transpose(np.matrix(z))
    return z


def py_xcov(a, b):


    a_r = np.array(a,dtype=np.double)  
    b_r = np.array(b,dtype=np.double)

        
    a_ = (a_r/len(a))
    b_ = (b_r/len(b))

    #Compute the mean
    a_mean = np.mean(a_)
    b_mean = np.mean(b_)
    

    a_ = a_ - a_mean
    b_ = b_ - b_mean


    #compare if the vectors a and b are of the same length, else append zeros to the end of the shorter vector!
    if (len(a) > len(b)):
        len_b = len(a) - len(b)
        b_=np.pad(b_,(0,len_b),'constant',constant_values=(0))
    elif (len(a) < len(b)):
        len_a = len(b) - len(a)
        a_=np.pad(a_,(0,len_a),'constant',constant_values=(0))
    
    #     else:
    #         they are of same length ;-)...


    ab_xcov = np.correlate(a_,b_,"full")
    return ab_xcov

# #Function equivalent of MatLab xcov...cross-covariance!
# def py_xcov(a, b):

#     a_ = np.array(a)
#     b_ = np.array(b)
#     a_ = a_/len(a_)
#     b_ = b_/len(b_)
#     a_ = a_-a_.mean()
#     b_ = b_-b_.mean()
    
    
#     #compare if the vectors a and b are of the same length, else append zeros to the end of the shorter vector!
#     if (len(a) > len(b)):
#         len_b = len(a) - len(b)
#         b_=np.pad(b_,(0,len_b),'constant')
#     elif (len(a) < len(b)):
#         len_a = len(b) - len(a)
#         a_=np.pad(a_,(0,len_a),'constant')
    
#     #     else:
#     #         they are of same length ;-)...

#     #Compute cross-correlation of mean-removed sequences to produce the 
#     z = np.correlate(a_, b_,"full")
#     #print(ab_xcov)
#     return z

# Time align adjustment functions to adjust gyro, acc meta data and values accordingly...
#S_acc = [S_acc_t, S_acc_tstamps, S_acc_counter, S_acc_values, S_acc_Ts]
def deleteFirstN(S,n):
    n = n+1
    S[0][0:n] = [] #S.t
    S[1][0:n] = [] #S.tstamps
    S[2][0:n] = [] #S.counter
    S[3] = np.delete(S[3], [*range(0,n)], axis=0)#S.values
    #S[3][0:n,:] = [] #S.values
    return S

def keepFirstN(S,n):
    S[0] = S[0][0:n] #S.t
    S[1] = S[1][0:n] #S.tstamps
    S[2] = S[2][0:n] #S.counter
    S[3] = S[3][0:n,:] #S.values
    return S

def timeAlign(S_a, S_g):
    # for peak location with sub-integer accuracy:
    neighborhood_half_width = 4
    z = py_xcov(S_a[0], S_g[0])
    # rough peak location:
    peak_val = max(z)
    kpeak = np.argmax(z)

    # peak location with sub-integer accuracy:
    ind = [*range(-neighborhood_half_width,neighborhood_half_width+1)]
    z_local = z[kpeak-neighborhood_half_width:kpeak+neighborhood_half_width+1]

    # fit a parabola
    P = np.polyfit(ind,np.double(z_local),2)
    # peak with sub-integer accuracy
    ind_local_peak = - 0.5*P[1]/P[0]
    peak_val = np.polyval(P,ind_local_peak)

    kpeak = kpeak + ind_local_peak


    # compute shift (round to nearest integer)
    shift = int(round(kpeak - (len(z) + 1)/2))
    
    
    # print(output)
    #display(md(output))
    
#     if shift == 1:
#         shift = 0

    if shift < -1:
        #print('deleting gyro values')
        S_g = deleteFirstN(S_g,shift)
    else:
        #print('deleting accel values')
        S_a = deleteFirstN(S_a,shift)

    if len(S_a[0]) > len(S_g[0]):
        #print('keeping accel values')
        S_a = keepFirstN(S_a,len(S_g[0]))
    else:
        #print('keeping gyro values')
        S_g = keepFirstN(S_g,len(S_a[0]))
        
    return S_a, S_g, shift


def readIMUSegments(urls):  
    n = len(urls)
    data = []
    ct = []
    tstamps = []
    S_t = []
    for k in range(n):
        C = urls[k].split("/")
        #print('GET segment:',C[len(C)-1])
        #r = requests.get(url = urls[k])
        r = (open_url(urls[k]).read())
        #decoded_content = r.content.decode('utf-8')
        cr = csv.reader(r.splitlines(), delimiter=',')
        my_list = list(cr)
        for row in my_list:
            e1 = int(row[0])
            ct.append(e1)
            e2 = np.double(row[1])
            e3 = np.double(row[2])
            e4 = np.double(row[3])
            e5 = datetime.datetime.strptime(row[4], '%Y-%m-%d %H:%M:%S.%f')
#             e5 = datetime.strptime(row[4], '%Y-%m-%d %H:%M:%S.%f')
            S_t.append(datetime_pytom(e5) * 24 * 3600)
            tstamps.append(e5)
            all_elements = [e1,e2,e3,e4,e5]
            data.append(all_elements)
            
    data = np.matrix(data)
            
#     a = data[:,0]

#     counter = a.tolist()
    c = len(ct)

#    print('\nwraparound counts at:')
#     ct = []
#     for i in range(c):
#         ct.append(int(counter[i][0]))
#    print(np.where(np.diff(ct, axis=0) != 1))
    counter = ct

#     timestamps = data[:,4]
#     ctstamps = timestamps.tolist()
#     cts = len(ctstamps)
#     tstamps = []
#     for i in range(cts):
#         tstamps.append(datetime.strptime(ctstamps[i][0], '%Y-%m-%d %H:%M:%S.%f'))
        
    #strg = '{:%Y-%m-%d %H:%M:%S.%f}'.format(ct[0]) #back from datetime format
    
    to = datetime_pytom(tstamps[0]) #time at the begining
    tf = datetime_pytom(tstamps[c-1]) #final time stamp recorded
    duration_in_seconds = (tf - to) * 24 * 3600
#     sec = str(duration_in_seconds.seconds) + '.' + str(duration_in_seconds.microseconds)
#     duration_in_seconds = float(sec)
    samples_per_second = (c - 1) / duration_in_seconds

    #to = datenum(tstamps(1),'yyyy-mmmm-dd HH:MM:SS.FFF');
    #tf = datenum(tstamps(end),'yyyy-mmmm-dd HH:MM:SS.FFF');

    #duration_in_seconds = (tf - to) * 24 * 3600;
    #samples_per_second = (size(tstamps,1) - 1) / duration_in_seconds;
    Ts = 1/samples_per_second
    time1 = '{:%Y-%m-%d %H:%M:%S.%f}'.format(tstamps[0])
    time2 = '{:%Y-%m-%d %H:%M:%S.%f}'.format(tstamps[c-1])

    #display(md(f'First timestamp: {time1}'))
    #display(md(f'Last timestamp : {time2}'));
    #display(md(f'<span style="font-family:Ariel;font-size:16px">Total number of samples: {c}</span>'))

    #display(md(f'<span style="font-family:Ariel;font-size:16px">Measured duration in seconds: {duration_in_seconds}</span>'));
    #display(md(f'<span style="font-family:Ariel;font-size:16px">Samples per second: {samples_per_second}</span>'));

    #S.t = (0:size(tstamps,1)-1)' * Ts;
    #S.t = (datenum(tstamps,'yyyy-mmmm-dd HH:MM:SS.FFF')- to) * 24 * 3600;
    #S_t =  date.toordinal(tstamps[0]) * 24 * 3600
    S_tstamps = tstamps
    S_counter = counter
    S_values = data[:,1:4]
    S_Ts = Ts
    
    return S_t, S_tstamps, S_counter, S_values, S_Ts

  
#------------------------------------------------------------------

#------------------------------------------------------------------

# vali_mes.value=''
# display(vali_mes)

#print(session_id)

#end_mes = widgets.Label('')

# add a widget later to read the capture session

def loadIMUData(event):

    # Check if the button is clicked
    # if not button1_clicked:
    #     # If not clicked, set the flag and return
    #     global button1_clicked
    #     button1_clicked = True
    #     return
        
    while not content_urls or not r_url:
        time.sleep(1)

    result = str(content_urls) + str(r_url)
    # print("Result from loadIMUData:", result)
    output_ta = document.querySelector("#output")
    enableLoad2Button()
    output_ta.innerText = result


def enableLoad2Button():
    document.getElementById('button2').disabled = False

    
# api-endpoint
pi = 3.1415926535897932384626433832795
session_id=1336
atmosURL = "https://xwjfurh8xk.execute-api.us-west-1.amazonaws.com/dev"
URL = atmosURL + "/capture_sessions/" + str(session_id) + "/content_urls"
req_url = atmosURL + "/capture_sessions/" + str(session_id)

# sending get request and saving the response as response object
#r_url = requests.get(url = req_url)
r_url = json.loads(open_url(req_url).read())

print("r_url",r_url) 

r_data = r_url
#r_data = str(r_data)
#r_data = r_data.replace("'", '"')

#Convert data to Python dict
request_url = r_data#json.loads(r_data)
cap_sess = copy.deepcopy(request_url)


# sending get request and saving the response as response object
#r = requests.get(url = URL)
r = json.loads(open_url(URL).read())

# extracting data in json format
data = r

content_urls = data


print("content_urls",content_urls) 

S_acc_t, S_acc_tstamps, S_acc_counter, S_acc_values, S_acc_Ts = readIMUSegments(content_urls['THETAX.acc']) 
S_acc_values = S_acc_values * 1000 / 9.81 #units of mg
S_acc = [S_acc_t, S_acc_tstamps, S_acc_counter, S_acc_values, S_acc_Ts]

#print(S_acc[3])

S_gyro_t, S_gyro_tstamps, S_gyro_counter, S_gyro_values, S_gyro_Ts = readIMUSegments(content_urls['THETAX.gyro']) 
S_gyro_values = S_gyro_values * 180 / pi #units of dps
S_gyro = [S_gyro_t, S_gyro_tstamps, S_gyro_counter, S_gyro_values, S_gyro_Ts]

for k in range(10):
    S_acc, S_gyro, shift = timeAlign(S_acc, S_gyro)
    if shift == -1:
        break
        
n = len(S_acc[0]);
estimated_duration = 0.5*(S_acc[0][len(S_acc[0])-1] + S_gyro[0][len(S_gyro[0])-1] - S_acc[0][0] - S_gyro[0][0])
samples_per_second =  (n-1) / estimated_duration
Ts = 1/samples_per_second

t_start_est = 0.5*(mean(S_acc[0]) + mean(S_gyro[0]) - estimated_duration)
dm = datetime_mtopy(t_start_est/24/3600)
start_time = '{:%Y-%m-%d %H:%M:%S.%f}'.format(dm)
#start_time = datestr(t_start_est/24/3600,'yyyy-mm-dd HH:MM:SS.FFF');


# % The beginning of the capture session is driven by the arrival of the
# % first *keyframe*. Which means that the IMU data can start *earlier*
# % than the capture session (sometimes by almost the keyframe interval).
# % When this happens time_offset is negative
# cap_sess_start = datenum(cap_sess.start_time, 'yyyy-mmmm-ddTHH:MM:SS.FFF') * 24 * 3600;
r_url_start_time = request_url['start_time']#.replace('T', ' ')
# r_url_start_time = r_url_start_time.replace('Z', '')
# st = datetime.datetime.strptime(r_url_start_time, '%Y-%m-%d %H:%M:%S.%f')
st = datetime.datetime.strptime(r_url_start_time, '%Y-%m-%dT%H:%M:%S.%fZ')
# cap_sess_start = round(datetime_pytom(st) * 24 * 3600)
cap_sess_start = datetime_pytom(st) * 24 * 3600
# t_offset = t_start_est - cap_sess_start
t_offset = round(t_start_est, 4) - cap_sess_start
#display(md(f'<span style="font-family:Ariel;font-size:14px">Time-offset: {t_offset}</span>'))

# % The end_time of the capture session is a time stamp generated after 
# % the IMU sensors stop. But if the end_time was recorded without msec
# % then it is possible for the IMU data to end *after* the capture
# % session.
# cap_sess_duration = datenum(cap_sess.end_time, 'yyyy-mmmm-ddTHH:MM:SS.FFF') * 24 * 3600 - cap_sess_start;
r_url_end_time = request_url['end_time']#.replace('T', ' ')
# r_url_end_time = r_url_end_time.replace('Z', '')
# et = datetime.datetime.strptime(r_url_end_time, '%Y-%m-%d %H:%M:%S.%f')
et = datetime.datetime.strptime(r_url_end_time, '%Y-%m-%dT%H:%M:%S.%fZ')
# cap_sess_duration = round(datetime_pytom(et) * 24 * 3600) - cap_sess_start
cap_sess_duration = (datetime_pytom(et) * 24 * 3600) - cap_sess_start

#print(S_gyro[3])
#Store the values to be used by other jupyter notebooks...


IMU_dt = Ts

IMU_start_time = start_time


IMU_capture_session = cap_sess

IMU_capture_session_duration = cap_sess_duration

IMU_time_offset = t_offset

#Session ID
#IMU_t = [0:n-1]' * IMU_dt --> in MatLab
IMU_t = (np.matrix(list(range(0,n)),dtype=np.double))  * IMU_dt
IMU_t = np.transpose(IMU_t)

IMU_acc_x = S_acc[3][:,0]
IMU_acc_y = S_acc[3][:,1]
IMU_acc_z = S_acc[3][:,2]


IMU_gyro_x = S_gyro[3][:,0]
IMU_gyro_y = S_gyro[3][:,1]
IMU_gyro_z = S_gyro[3][:,2]

# with open('shared_data.txt', 'w') as file:
#     file.write(f"IMU_acc_x = {IMU_acc_x.tolist()}\n")
#     file.write(f"IMU_acc_y = {IMU_acc_y.tolist()}\n")
#     file.write(f"IMU_acc_z = {IMU_acc_z.tolist()}\n")
#     file.write(f"IMU_gyro_x = {IMU_gyro_x.tolist()}\n")
#     file.write(f"IMU_gyro_y = {IMU_gyro_y.tolist()}\n")
#     file.write(f"IMU_gyro_z = {IMU_gyro_z.tolist()}\n")
#     file.write(f"IMU_dt = {IMU_dt}\n")

with open('shared_data.txt', 'w') as file:
    file.write(f"IMU_acc_x = {IMU_acc_x.tolist()}\n")
    file.write(f"IMU_acc_y = {IMU_acc_y.tolist()}\n")
    file.write(f"IMU_acc_z = {IMU_acc_z.tolist()}\n")
    file.write(f"IMU_gyro_x = {IMU_gyro_x.tolist()}\n")
    file.write(f"IMU_gyro_y = {IMU_gyro_y.tolist()}\n")
    file.write(f"IMU_gyro_z = {IMU_gyro_z.tolist()}\n")
    file.write(f"IMU_dt = {IMU_dt}\n")

# print(IMU_acc_x[1]) 

# while not content_urls or not r_url:
#     time.sleep(1)

# js.document.getElementById('output').value = str(content_urls) + str(r_url)

  

# output_data = {
#     'IMU_acc_x': IMU_acc_x.tolist(),
#     'IMU_acc_y': IMU_acc_y.tolist(),
#     'IMU_acc_z': IMU_acc_z.tolist(),
#     'IMU_gyro_x': IMU_gyro_x.tolist(),
#     'IMU_gyro_y': IMU_gyro_y.tolist(),
#     'IMU_gyro_z': IMU_gyro_z.tolist()
# }

# # Convert the Python list to a JavaScript array
# js_output = json.dumps(output_data)

# # Use PyScript to execute JavaScript code in the browser
# js_code = f"document.getElementById('output').value = {js_output};"
# window.exec(js_code)


#--------------------------------------------------------------------
