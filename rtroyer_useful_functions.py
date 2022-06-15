#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:08:25 2020

@author: rntroyer
"""

import numpy as np
import os
import datetime
from skimage import exposure
from skimage.restoration import denoise_nl_means, estimate_sigma
from IPython.display import clear_output
import smtplib
import ssl
import ftplib
import wget
from bs4 import BeautifulSoup
import requests
import h5py


def get_url_paths(url, ext='', params={}):
    """ Function to extract file names from https directory
    Gets files in url directory with ext extension
    Does this by parsing the html text from the webpage. I did not
    write this function. Requires libraries requests,
    bs4.BeautifulSoup
    DEPENDENCIES
        bs4.BeautifulSoup, requests
    INPUT
    url
        type: string
        about: url of directory to get files from
    ext
        type: string
        about: extension of the files
    OUTPUT
    parent
        type: list
        about: list of all file pathnames within directory
    """

    response = requests.get(url, params=params)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()
    soup = BeautifulSoup(response_text, 'html.parser')
    parent = [url + node.get('href') for node in soup.find_all('a')
              if node.get('href').endswith(ext)]
    return parent

def datetime_arange(start_dt, end_dt, milliseconds):
    """Function to create a numpy arange of datetimes, mainly used
    to set timestamps for data.
    INPUT
    start_dt, end_dt
        type: datetime
        description: start and end times for range
    milliseconds
        type: float
        description: the spacing between timestamps
    OUTPUT
    dtstamps
        type: numpy array of datetimes
        description: the timestamp values
    """
    dtstamps = np.arange(start_dt, end_dt, 
                         datetime.timedelta(milliseconds=milliseconds)
                        ).astype(datetime.datetime)
    return dtstamps


def img_clean_n_boost(img, weight=1, low=0.1, high=99.9):
    """Function to denoise and boost the contrast of an image.
    Primarily built for auroral images. Uses denoise_nl_means
    with sigma_estimate from skimage.restoration and rescale_intensity
    from skimage.io.exposure.
    INPUT
    img
        type: 2D array
        description: array representing the image to be processed
    weight = 1
        type: float
        description: value that defines how much denoising to do
    low, high
        type: float
        description: values that define the contrast range
    OUTPUT
    processed_img
        type: 2D array
        description: array representing the processed image
    """
    
    # Denoise first
    #...by estimating the sigma of the image for noise
    sigma_est = np.mean(estimate_sigma(img, multichannel=True))

    #...and keywords to specify the patch size for the denoise algorithm
    patch_kw = dict(patch_size=5, patch_distance=6, multichannel=True)
    #...then perform the denoising
    processed_img = denoise_nl_means(img, h=weight*sigma_est,
                                     sigma=sigma_est,
                                     fast_mode=True, **patch_kw)

    # Boost the contrast second
    v_min, v_max = np.percentile(processed_img, (low, high))
    processed_img = exposure.rescale_intensity(processed_img,
                                               in_range=(v_min, v_max))
    
    return processed_img

def update_progress(progress, process=''):
    """Function to display a bar with the progress of an action.
    INPUT
    progress
        type: float
        description: action step/total steps of action
    process
	type: string
	about: optional string to add to string printout
    OUTPUT
    N/A
    """
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progress (" + process + "): [{0}] {1:.1f}%".format( "#" * block 
                                             + "-" * (bar_length - block),
                                             progress * 100)
    print(text)

def mask_image(image, x_shift, y_shift, radius, 
               edge=0):
    """ Function is designed to blacken an allsky image outside
    of the sky region. Input the radius (in pixels) of the sky section
    and the pixel coordinates of the center of the sky section.
    Returnsa masked image. Also has the ability to mask edges 
    with pixel width w.
    INPUT
    image
        type: array
        about: the image array to mask
    x_shift
     type: int
     about: number of pixels to shift the center in x-direction
    y_shift 
        type: int
        about: number of pixels to shift the center in y-direction
    radius 
        type: int
        about: radius in pixels of the circle to mask image with
    edge 
        type: int
        about: pixels to mask out around the image edge
    OUTPUT
    image 
        type: array
        about: the masked image array with masked values as nan
    """
    
    # First convert image to float
    image = image.astype('float')
    
    # Define parameters for mask all in pixels
    x_center = int(image.shape[1]/2) + x_shift 
    y_center = int(image.shape[0]/2) + y_shift
    
    # Create an array with shifted index values
    x_array = np.zeros([image.shape[0], image.shape[1]], dtype=int)
    for i in range(0, x_array.shape[0]):
        x_array[i,:] = np.arange(0, image.shape[1])

    y_array = np.zeros([image.shape[0], image.shape[1]], dtype=int)
    for j in range(0, y_array.shape[1]):
        y_array[:,j] = np.arange(0, image.shape[0])

    # Shift the arrays
    x_array = x_array - x_center
    y_array = y_array - y_center

    # Array to store radius value
    radius_array = x_array**2 + y_array**2

    # Set image array to black outside of radius
    black = np.nan
    image[np.where(radius_array > radius**2)] = black
    
    # Edge masking
    image[:edge+1, :] = black
    image[:, :edge+1] = black
    image[image.shape[1] - (edge+1):, :] = black
    image[:, image.shape[1] - (edge+1):] = black
    
    return image        

def get_isr_data(pfisr_filename, pfisr_data_dir):
    """Function to get relevant data from PFISR datafile.
    INPUT
    pfisr_filename
        type: str
        about: data file name, should be .h5 file
    pfisr_data_dir
        type: str
        about: directory where isr data is stored
    OUTPUT
    utc_time
        type: array of datetimes
        about: time stamp for the start of each measurement
    unix_time
        type: array of floats
        about: unix timestamp for start of each measurement
    pfisr_altitude
        type: array of float
        about: altitude stamp for each measurement in meters
    e_density
        type: array of float
        about: electron number density in m^-3
    de_density
        type: array of float
        about: error in number density
    """
    
    # Read in the h5 file
    pfisr_file = h5py.File(pfisr_data_dir + pfisr_filename, 'r')

    # Get the different beams and select specified angle
    beam_angle = 90
    beams = np.array(pfisr_file['BeamCodes'])

    # Get the beam with a 90 degree elevation angle
    indexes = np.linspace(0, len(beams)-1, len(beams))
    beam_num = int(indexes[np.abs(beams[:,2] - beam_angle) == 0][0])

    # Get time and convert to utc datetime
    unix_time = np.array(pfisr_file['Time']['UnixTime'])[:,0]
    utc_time = np.array([datetime.datetime.utcfromtimestamp(d) 
                         for d in unix_time])

    # Get the altitude array
    pfisr_altitude = np.array(pfisr_file['NeFromPower']
                              ['Altitude'])[beam_num, :]

    # Get the uncorrected number density array
    e_density = np.array(pfisr_file['NeFromPower']
                         ['Ne_NoTr'])[:, beam_num, :]

    # Take the transpose
    e_density = np.transpose(e_density)
    
    # Find the noise floor by averaging between 55km and 60km
    #...assume this should be zero
    
    # Calculate the power given that power = density/range^2
    pfisr_range = np.array(pfisr_file['NeFromPower']
                           ['Range'])[0, :]

    # Turn 1D array into 2D array for elementwise division
    pfisr_range = np.array([pfisr_range,]*e_density.shape[1])
    pfisr_range = np.transpose(pfisr_range)
    pfisr_power = np.divide(e_density, pfisr_range**2)

    # Get the power bias
    noise_floor = np.nanmean(pfisr_power[(pfisr_altitude > 55000)
                                    & (pfisr_altitude < 60000), :],
                              axis=0)

    # Loop through each column and subtract off noise floor
    for j in range(pfisr_power.shape[1]):
        pfisr_power[:, j] = pfisr_power[:, j] - noise_floor[j]   

    # Calculate new unbiased density
    e_density = np.multiply(pfisr_power, pfisr_range**2)
        
    
    # Get error values
    try:
        de_density = np.array(pfisr_file['NeFromPower']
                              ['errNe_NoTr'])[:, beam_num, :]
        de_density = np.transpose(de_density)
    except:
        de_density = np.array(pfisr_file['NeFromPower']
                              ['dNeFrac'])[:, beam_num, :]
        de_density = np.transpose(de_density)
        de_density = de_density * e_density

    # Close file
    pfisr_file.close()
    
    return utc_time, unix_time, pfisr_altitude, e_density, de_density


def reduced_chi_squared(observed, modeled, errors):
    """Function to calculate the chi square test for PFISR data.
    DEPENDENCIES
        numpy
    INPUT
    observed
        type: array
        about: observed values that model is trying to fit
    modeled
        type: array
        about: model that is attempting to fit observed data
    errors
        type: array
        about: errors in observed values
    OUTPUT
    chi_square / (len(observed) - 1)
        type: float
        about: reduced chi squared value 
                << 1 over fit >> 1 not fit well
    """
    
    # Squared difference between observed and modeled data
    model_diff_sq = (observed - modeled)**2
    
    # Variance or errors squared
    variance_sq = errors**2
    
    # Calculate chi square array by dividing differences by error
    chi_square_array = model_diff_sq/variance_sq
    
    # Sum to get value
    chi_square = np.sum(chi_square_array)
    
    # Reduce chi square by dividing by number of points -1 
    return chi_square / (len(observed) - 1)
