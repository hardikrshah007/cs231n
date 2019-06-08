#!/usr/bin/python3

import argparse
import numpy
import matplotlib.pyplot as plt
import ipdb
import os
from matplotlib.backends.backend_pdf import PdfPages
import math
from scipy.optimize import least_squares
import pyquaternion
from scipy.optimize import curve_fit

# Below function is for doing a linear fit on data
def f(x,A,B):
	return A*x+B

def angle(homo): #This function return tan-1 in radians
    return math.atan2(homo[1,0], homo[0,0])
# Below function is to convert a 3X3 matrix into x,y,Theta
def to_state(homo):
    state = numpy.zeros((3,))
    state[0] = homo[0,2]
    state[1] = homo[1,2]
    state[2] = angle(homo)

    return state

def eye(n=3):
    return numpy.matrix(numpy.eye(n))

# Below function is to convert x,y,theta into 3x3 matrix
def to_homo(state):
    state = state.flatten() # this coverts the array into 1 D
    homo = eye()
    homo[0,0] = math.cos(state[2])
    homo[0,1] = -math.sin(state[2])
    homo[1,0] = math.sin(state[2])
    homo[1,1] = math.cos(state[2])
    homo[0,2] = state[0]
    homo[1,2] = state[1]
    return homo

def projxz_3d_to_2d(left_expressedIn_left0):
# , left0_expressedIn_velometer):

    # left0 (left camera origin)
    #    ------ z (z=camera lens axis)
    #   /|
    #  / |
    # x  y

    # left,C (if the camera is mounted upside down on the agv)
    #  yC  xC
    #  | /
    #  |/
    #  ------ zC (z=camera lens axis)

    # A (the velometer-like frame at the camera)
    #  zA  yA
    #  | /
    #  |/
    #  ------ xA (x=camera lens axis)

    M = left_expressedIn_left0
    
    xaxis = numpy.copy(numpy.squeeze(numpy.asarray(M[0:3,0])))
    xaxis[1] = 0.
    if numpy.linalg.norm(xaxis) < 0.25:
        raise "transform is not sufficiently planar to calculate heading"

    if args.upsidedown:
        # the "angle" we use for A is the angle of xC wrt to left0 frame
        # and we want it to be about zA (so positive axis up)
        angle = math.atan2(-xaxis[2], -xaxis[0]) # angle=atan(-z/-x)
        state = numpy.array([M[2,3], -M[0,3], angle]) #+x along camera axis (xA)
    else:
        angle = math.atan2(xaxis[2], xaxis[0]) # angle=atan(z/x)
        state = numpy.array([M[2,3], -M[0,3], angle]) #+x along camera axis
        
    return to_homo(state)


def accumulate_incremental_poses(states, current_expressedIn_world=eye()):

    current_expressedIn_world = numpy.copy(current_expressedIn_world)

    out_states = numpy.zeros((states.shape[0], 3))

    #current_expressedIn_world = eye()
    out_states[0,:] = to_state(current_expressedIn_world)

    for r in range(states.shape[0]-1):
        current_expressedIn_prev = to_homo(states[r,:])
        current_expressedIn_world = current_expressedIn_world \
                                    * current_expressedIn_prev
        out_states[r+1,:] = to_state(current_expressedIn_world)

      
    return out_states


def nearest_idx(needle, haystack):
    # find the index of the value nearest data
    return numpy.abs(haystack-needle).argmin()

def process_cam_poses(data):
# ,left_expressedIn_velometer):

    N = data.shape[0]

    left_expressedIn_world = numpy.zeros((N,3))
    left3d_expressedIn_world = numpy.zeros((4,4,N))

    for r in range(N):
        a = data[r,1:]
        M = eye(4)
        M[0:3,:] = numpy.reshape(a, (3,4))
        M = numpy.linalg.inv(M)

        # enforce orthogonality caused by floating
        # point rounding in txt
        [U,S,Vt] = numpy.linalg.svd(M[0:3,0:3])
        M[0:3,0:3] = U*Vt
        n1 = M.shape[0]
        n2 = M.shape[1]

        left_expressedIn_world[r,:] = to_state(projxz_3d_to_2d(M))

        if abs(numpy.linalg.det(M[0:3,0:3]) - 1.) > 1e-6:
            ipdb.set_trace()
            print("warning: bad determinant")

        left3d_expressedIn_world[:,:,r] = M

    return (left_expressedIn_world, left3d_expressedIn_world)


def slerp(f,
          from_expressedIn_other,
          to_expressedIn_other):
    to_expressedIn_from = numpy.linalg.inv(from_expressedIn_other) \
                          * to_expressedIn_other

    a = to_state(to_expressedIn_from)
    slerped_expressedIn_from = to_homo(a * f)

    return from_expressedIn_other * slerped_expressedIn_from
    
def interp_abs(desiredTime,
               times,
               obj_expressedIn_world):

    assert(times.shape[0] == obj_expressedIn_world.shape[0])

    idx = nearest_idx(desiredTime, times)

    if desiredTime <= times[idx]:
        if idx > 0:
            # times[idx0] < desiredTime <= times[idx1]
            idx0 = idx-1
            idx1 = idx
        else:
            # extrapolation before beginning
            idx0 = 0
            idx1 = 1
            #ipdb.set_trace()
    else:
        # desiredTime > times[idx]
        lastIdx = times.shape[0]-1
        if idx < lastIdx:
            # times[idx0] < desiredTime <= times[idx1]
            idx0 = idx
            idx1 = idx+1
        else:
            # extrapoation past ending
            idx0 = lastIdx-1
            idx1 = lastIdx
            #ipdb.set_trace()
            
    f = (desiredTime - times[idx0]) / (times[idx1] - times[idx0])

    
    return slerp(f,
                 to_homo(obj_expressedIn_world[idx0,:]),
                 to_homo(obj_expressedIn_world[idx1,:]))

def resample_incr_from_abs(timeVelometer,
                           velometerTimeOffset,
                           timeLeft, left_expressedIn_world):

    N = timeVelometer.shape[0]
    assert(timeLeft.shape[0] == left_expressedIn_world.shape[0])

    incr_states = numpy.zeros((N,3))

    for r in range(N-1):

        startLeft_expressedIn_world = interp_abs(timeVelometer[r]+velometerTimeOffset,
                                                 timeLeft,
                                                 left_expressedIn_world)
        stopLeft_expressedIn_world = interp_abs(timeVelometer[r+1]+velometerTimeOffset,
                                                timeLeft,
                                                left_expressedIn_world)
        
        stopLeft_expressedIn_startLeft \
            = numpy.linalg.inv(startLeft_expressedIn_world) \
            * stopLeft_expressedIn_world

        incr_states[r+1,:] = to_state(stopLeft_expressedIn_startLeft)

    return incr_states

def inv(s):
    return to_state(numpy.linalg.inv(to_homo(s)))
    
def transform_incr_states(currentLeft_expressedIn_prevLeft,
                          velometer_expressedIn_left):
    N = currentLeft_expressedIn_prevLeft.shape[0]
    currentLeft_expressedIn_prevLeft_velometer = numpy.zeros((N,3))
    C = to_homo(velometer_expressedIn_left)
    invC = numpy.linalg.inv(C)

    for r in range(N):
        M = to_homo(currentLeft_expressedIn_prevLeft[r,:])
        
        currentLeft_expressedIn_prevLeft_velometer[r,:] = to_state(invC * M * C)

    return currentLeft_expressedIn_prevLeft_velometer

def transform_states(camerapt_expressedIn_velometer,
                     velometer_expressedIn_2dworld):
    N = camerapt_expressedIn_velometer.shape[0]
    camerapt_expressedIn_2dworld = numpy.zeros((N,3))
    
    C = to_homo(velometer_expressedIn_2dworld)
    
    for r in range(N):
        M = to_homo(camerapt_expressedIn_velometer[r,:])
        # velometer_expressedIn_world[r,:] = to_state(M * C)
        camerapt_expressedIn_2dworld[r,:] = to_state(M * C)
    return camerapt_expressedIn_2dworld

def cost(velometer_expressedIn_left,
         currentLeft_expressedIn_prevLeft,
         currentVelometer_expressedIn_prev_velometer):

    currentOffsetLeft_expressedIn_prevOffsetLeft \
        = transform_incr_states(currentLeft_expressedIn_prevLeft,
                                velometer_expressedIn_left)

    err = currentOffsetLeft_expressedIn_prevOffsetLeft \
          - currentVelometer_expressedIn_prev_velometer
    # print(" The error in theta is: ", err[3])

    return numpy.reshape(err, (-1))


def RtoEuler(R):
    #angles(1)=roll angles(2)=pitch angles(3)=yaw
    R = numpy.squeeze(numpy.asarray(R.flatten('C'))) # row major
    epsi = 1e-5;
    sx = R[7];
    if (sx < -1.0 and sx > -1.0 - epsi):
        sx = -1.0;
    if(sx > 1.0 and sx < 1.0 + epsi):
        sx = 1.0;

    x_angle = math.asin(sx);
    cxabs = abs(math.cos(x_angle));
    if(cxabs < epsi):
        yplusz = math.atan2(R[3] + R[2],R[0] - R[5]);
        yminusz = math.atan2( R[2] - R[3], R[0] + R[5]);
        y_angle = (yplusz+yminusz)/2;
        z_angle = yplusz-y_angle;
    else:
        z_angle = math.atan2( -R[1], R[4]);
        y_angle = math.atan2( -R[6], R[8]);

    return (-z_angle, -x_angle, -y_angle)

def print_motion_estimation_calib(velometer_expressedIn_left2d,
                                  name,
                                  gravityAlignedLeft_expressedIn_left):
    # L = velometer
    # A = pretend left cam frame with +x forward, +z up
    # C = real-left-cam

    # left0 (left camera origin)
    #    ------ z (z=camera lens axis)
    #   /|
    #  / |
    # x  y

    # left,C (the camera is mounted upside down on the agv)
    #  y  x
    #  | /
    #  |/
    #  ------ z (z=camera lens axis)

    # A (the velometer-like frame at the camera)
    #  z  y
    #  | /
    #  |/
    #  ------ x (x=camera lens axis)

    # velometer
    #  z  y
    #  | /
    #  |/
    #  ------ x (the blind spot is along -x) 

    L_expressedIn_A = eye(4)
    L_expressedIn_A[0:2,0:2] = velometer_expressedIn_left2d[0:2,0:2]
    L_expressedIn_A[0:2,3] = velometer_expressedIn_left2d[0:2,2]

    C_expressedIn_A = eye(4);

    if args.upsidedown:
        # when upside down:
        C_expressedIn_A[0:3,0:3] = numpy.matrix([[0,0,1],[1,0,0],[0,1,0]])
    else:
        # when not upside down
        C_expressedIn_A[0:3,0:3] = numpy.matrix([[0,0,1],[-1,0,0],[0,-1,0]])

    L_expressedIn_C = numpy.linalg.inv(C_expressedIn_A) * L_expressedIn_A

    velometer_expressedIn_gravityAlignedLeft = L_expressedIn_C
    velometer_expressedIn_left = gravityAlignedLeft_expressedIn_left \
                             * velometer_expressedIn_gravityAlignedLeft

    M = velometer_expressedIn_left
    print("velometer2camera_translation_%s = %0.5f %0.5f %0.5f"
          % (name,
             M[0,3],
             M[1,3],
             M[2,3]))
    euler = RtoEuler(M[0:3,0:3]) 
    print("velometer2camera_rotation_%s = %0.10f %0.10f %0.10f"
          % (name,
             euler[0]/ math.pi * 180.,
             euler[1]/ math.pi * 180.,
             euler[2]/ math.pi * 180.))

def analyze_left_deviation_from_gravity(left3d_expressedIn_world):
    N = left3d_expressedIn_world.shape[2]
    angles = numpy.zeros((N,4))
    for i in range(N):

        left_expressedIn_world = numpy.matrix(left3d_expressedIn_world[:,:,i])
        if abs(numpy.linalg.det(left_expressedIn_world[0:3,0:3]) - 1.) > 1e-6:
            ipdb.set_trace()
               
        gravityAlignedLeft_expressedIn_world = eye(4)

        # match position
        gravityAlignedLeft_expressedIn_world[:,3] = left_expressedIn_world[:,3]

        # align such that +y is gravity (down) and +z is along original +z
        z = numpy.copy(left_expressedIn_world[0:3,2])
        z[1] = 0.
        z = z / numpy.linalg.norm(z)
        y = numpy.matrix([[0],
                          [math.copysign(1., left_expressedIn_world[1,1])],
                          [0]])
        
        x = numpy.cross(y.transpose(),z.transpose()).transpose()
        gravityAlignedLeft_expressedIn_world[0:3,0] = x
        gravityAlignedLeft_expressedIn_world[0:3,1] = y
        gravityAlignedLeft_expressedIn_world[0:3,2] = z

        p = pyquaternion.Quaternion(matrix=gravityAlignedLeft_expressedIn_world[0:3,0:3])
        q = pyquaternion.Quaternion(matrix=left_expressedIn_world[0:3,0:3])        
        log = pyquaternion.Quaternion.log_map(q,p)
        
        angles[i,0] = log[0]        
        angles[i,1] = log[1]
        angles[i,2] = log[2]
        angles[i,3] = log[3]

    mean_log = numpy.mean(angles, axis=0)

    gravityAlignedLeft_expressedIn_left \
        = pyquaternion.Quaternion.exp_map(pyquaternion.Quaternion(),
                                          pyquaternion.Quaternion(mean_log))

    print("gravityAlignedLeft_expressedIn_left=",
          gravityAlignedLeft_expressedIn_left.transformation_matrix)

    return (numpy.matrix(gravityAlignedLeft_expressedIn_left.transformation_matrix),
            angles)
    

def convertVelocity2distance(velometerTime,velocity_expressedIn_previousVelocities):

    N = velocity_expressedIn_previousVelocities.shape[0]
    N1 = velocity_expressedIn_previousVelocities.shape[1]
    deltaTime = numpy.zeros((N))
    distance_expressedIn_previousVelocities = numpy.zeros((N,N1))
    for i in range(N-1):
                deltaTime[i] = velometerTime[i+1] \
                    -velometerTime[i]
                distance_expressedIn_previousVelocities[i+1,:] =deltaTime[i]*velocity_expressedIn_previousVelocities[i,:]
        # print("Delta theta in degreesfor each time step is: ", distance_expressedIn_previousVelocities[i,-1])
    distance_expressedIn_previousVelocities[-1,:]= 0
    distance_expressedIn_previousVelocities[:,-1]= distance_expressedIn_previousVelocities[:,-1]*math.pi/180;
    # print("The theta column is: ")
    return (distance_expressedIn_previousVelocities)

def convert2Mat(velocity_expressedIn_previousVelocities):

    N = velocity_expressedIn_previousVelocities.shape[0]
    process_velometer = numpy.zeros((N,3))
    process_velometer[:,0] = velocity_expressedIn_previousVelocities[:,0]
    process_velometer[:,1] = velocity_expressedIn_previousVelocities[:,1]
    process_velometer[:,2] = velocity_expressedIn_previousVelocities[:,5]

    return (process_velometer)

def interpolation_abs_in_world(desiredTime,
               times,
               obj_expressedIn_world):

    assert(times.shape[0] == obj_expressedIn_world.shape[0])

    idx = nearest_idx(desiredTime, times)

    if desiredTime <= times[idx]:
        if idx > 0:
            # times[idx0] < desiredTime <= times[idx1]
            idx0 = idx-1
            idx1 = idx
        else:
            # extrapolation before beginning
            idx0 = 0
            idx1 = 1
            #ipdb.set_trace()
    else:
        # desiredTime > times[idx]
        lastIdx = times.shape[0]-1
        if idx < lastIdx:
            # times[idx0] < desiredTime <= times[idx1]
            idx0 = idx
            idx1 = idx+1
        else:
            # extrapoation past ending
            idx0 = lastIdx-1
            idx1 = lastIdx
            #ipdb.set_trace()
            
    f = (desiredTime - times[idx0]) / (times[idx1] - times[idx0])
    interpolated_point= obj_expressedIn_world[idx0,:] + f*(obj_expressedIn_world[idx1,:]-obj_expressedIn_world[idx0,:])
    return interpolated_point

def sample_from_abs(time_sampled,time_data,position_expressedIn_world):
    N = time_sampled.shape[0]
    new_sampled_positionIn_world= numpy.zeros((N,3))
    for i in range(N-1):
        new_sampled_positionIn_world[i,:]= interpolation_abs_in_world(time_sampled[i],time_data,position_expressedIn_world)
    return (new_sampled_positionIn_world)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="plot various results from velometermatching in 2d",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # parser.add_argument('results_dir', type=str,
    #                     help='DS_*** directory')
    # parser.add_argument('--upsidedown', action='store_true',
    #                     help='if the unit is mounted upside down')
    # parser.add_argument('--no_gravity_align', action='store_true',
    #                     help='calculate, but do not use, the gravity alignment')
 
    # args = parser.parse_args()
    lArray=[]
    l1=(1,2,3)
    # lsplit=numpy.array_split(lArray,5)
    lArray.extend(l1)
    l2=(4,5,6)
    lArray.extend(l2)
    # l2split=numpy.array_split(l1split,4)
    print("The first array is:",len(lArray))