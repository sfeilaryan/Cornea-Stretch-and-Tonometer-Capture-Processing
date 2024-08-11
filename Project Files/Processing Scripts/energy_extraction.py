## loading python libraries

# load the libraries
import matplotlib.pyplot as plt  # 2D plotting library
import numpy as np  # type: ignore # package for scientific computing

from scipy import optimize  # Numerically solve non-linear system of equations

from math import *  # package for mathematics (pi, arctan, sqrt, factorial ...)

import cv2  # In case I have to write code similar to that of Wu Yifan
import os
import preprocessing  # Wu Yifan's image/video processing functions
from PIL import Image  # Movie creation
import random
import global_variables as system_constants

def get_segment_lengths(x_0, y_0):  # Returns the N+1 segment Lengths, take N+2
    # node coordinates
    lengths = np.zeros(system_constants.N + 1)
    for i in range(x_0.shape[0] - 1):
        length = sqrt((x_0[i] - x_0[i + 1]) ** 2 + (y_0[i] - y_0[i + 1]) ** 2)
        lengths[i] = length
    return lengths


def slope(xy1, xy2):
    return (xy2[1] - xy1[1]) / (xy2[0] - xy1[0])

def place_x_nodes(interval, n_nodes):
    start, end = interval
    if n_nodes == 1:
        return [start]
    elif n_nodes == 2:
        return [start, end]
    
    step = (end - start) // (n_nodes - 1)
    
    # Generate nodes
    nodes = [start + i * step for i in range(n_nodes)]
    
    # Ensure the last element is exactly the end
    nodes[-1] = end
    
    return np.array(nodes)

def get_upper_waveform(videoName, imageNumber):
    imagePath = (
        "Data/video" + videoName + "_TREATED/" + "image" + str(imageNumber) + ".jpg"
    )
    frame = cv2.imread(imagePath)
    frame = preprocessing.image_processing(
        frame
    ) 
    waveform = preprocessing.get_waveform(frame)  # Upper and Lower surfaces
    upper_surface = waveform[:, 0]
    return upper_surface

def segment_energy(x_ext, y_ext, initial_length, stiffness, x_nodes, y_nodes):
    total_length = 0
    current_point = (x_ext[0], y_ext[0])
    end_point = (x_ext[1], y_ext[1])
    for i in range(x_nodes.shape[0]):
        if current_point[0] <= x_nodes[i] <= end_point[0]:
            total_length += np.linalg.norm(current_point- [x_nodes[i], y_nodes[i]])
            current_point = [x_nodes[i], y_nodes[i]]
        if x_nodes[i] > end_point[0]:
            break
    total_length += np.linalg.norm(current_point- end_point)
    return 0.5 * stiffness * ((initial_length - total_length)**2)

def interpolate(x, y, u):
    # Extrapolation for u less than x[0]
    if u < x[0]:
        x_i, x_ip1 = x[0], x[1]
        y_i, y_ip1 = y[0], y[1]
        v = y_i + ((u - x_i) * (y_ip1 - y_i)) / (x_ip1 - x_i)
        return v

    # Extrapolation for u greater than x[-1]
    if u > x[-1]:
        x_i, x_ip1 = x[-2], x[-1]
        y_i, y_ip1 = y[-2], y[-1]
        v = y_i + ((u - x_i) * (y_ip1 - y_i)) / (x_ip1 - x_i)
        return v

    # Interpolation for u within the bounds
    for i in range(len(x) - 1):
        if x[i] <= u <= x[i + 1]:
            x_i, x_ip1 = x[i], x[i + 1]
            y_i, y_ip1 = y[i], y[i + 1]
            v = y_i + ((u - x_i) * (y_ip1 - y_i)) / (x_ip1 - x_i)
            return v
    
    # If the code reaches here, it means something went wrong
    raise RuntimeError("Interpolation error")

def potential_energy(u_arr, x, y, lengths, k):
    u = np.zeros(x.shape[0])
    for j in range(u_arr.shape[0]):
        u[j+1] = u_arr[j] 
    new_x = x + u
    new_y = np.zeros(y.shape[0])
    for i in range(y.shape[0]):
        new_y[i] = interpolate(x, y, new_x[i])
    segment_energies = np.zeros(k.shape[0])
    for l in range(segment_energies.shape[0]):
        segment_energies[l] = segment_energy(new_x[l:l+2], new_y[l:l+2], lengths[l], k[l], x, y)
    energy = np.sum(segment_energies)
    return energy

def get_frame_potential(video_name, image_number, initial_lengths, K_arr, initial_guess, x_nodes):
    waveform = get_upper_waveform(video_name, image_number)
    y_nodes = waveform[x_nodes]
    disp = optimize.minimize(
        potential_energy,
        initial_guess,
        args = (x_nodes, y_nodes, initial_lengths, K_arr)
        )
    displacements = disp.x
    energy = potential_energy(displacements, x_nodes, y_nodes, initial_lengths, K_arr)
    return energy, displacements


def potential_evolution(
    video_name, total_images, increment = 1
):  # returns Potential Array (Chronological)
    image_number = 1
    x_nodes = place_x_nodes((0, 575), system_constants.N +  2)
    waveform = get_upper_waveform(video_name, image_number)
    y_nodes = waveform[x_nodes]
    initial_lengths = get_segment_lengths(x_nodes, y_nodes)
    #print(initial_lengths)
    #print(f'Initial lengths: {initial_lengths}')
    stiffnesses = system_constants.K/(initial_lengths)
    energy_array = np.zeros(total_images)
    initial_guess = np.zeros(system_constants.N)
    displacements = []
    while image_number <= total_images:
       print(f'Processing Frame: {image_number}  ')
       new_energy, displacement = get_frame_potential(video_name,
                                                       image_number,
                                                       initial_lengths,
                                                       stiffnesses,
                                                       initial_guess,
                                                       x_nodes)
       energy_array[image_number - 1] = new_energy
       print(f'Frame Energy = {new_energy}')
       displacements.append(displacement)
       initial_guess = displacement
       image_number += increment
    return energy_array, np.array(displacements)
