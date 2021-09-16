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

def send_email(message, password=None, receiver_email=None):
    """Function to send an email notification.
    INPUT
    message
        type: str
        about: Should be in form of Subject: subject \n
               message.
    password=None
        type: str
        about: password to sender account
   receiver_email=None
        type: string
        about: email address to send email to
    OUTPUT
    none
    """
    #...sending email
    sender_email = 'riley.troyer.python@gmail.com'
    #...password to google account
    password = 'Aurora557'
    #...receiving email
    receiver_email = 'riley.troyer94@icloud.com'
    #...connect to this port, required for Gmail
    port = 465

    # Create a secure SSL context
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com',
                          port, context=context) as server:
        # Login
        server.login('riley.troyer.python@gmail.com', password)
        #...then send message
        server.sendmail(sender_email, receiver_email, message)

def read_pfisr_file(date_str, 
                    data_dir=('../data/pfisr-data/mswinds/')):

    """Function to read a pfisr data file and return the parameters 
    as numpy arrays. Requires data file to be downloaded.
    INPUT
    DEPENDENCIES
        os, h5py, numpy
    date_str
        type: string
        about: string with date to read file for, YYYYMMDD
    OUTPUT
    pfisr_time
        type: array of datetimes
        about: times for each pfisr measurement
    pfisr_alt
        type: array of floats
        about: altitudes for each pfisr measurement in kilometers
    e_density
        type: array of floats
        about: electron densities for each altitude and time (1/m^3)
    de_density
        type: array of floats
        about: error for each electron density measurement
    beam_az, beam_el
        type: array of floats
        about: azimuthal and elevation angles for all beams
    """
    # Read in the original pfisr file
    pfisr_files = os.listdir(data_dir)
    pfisr_files = [f for f in pfisr_files if f.endswith('.h5')]
    pfisr_file = [data_dir + f for f in pfisr_files if date_str in f][0]

    # Read in h5 file
    pfisr_file = h5py.File(pfisr_file, 'r')

    # Get the different beams and select specified angle
    beam_angle = 90
    beams = np.array(pfisr_file['BeamCodes'])

    # Get the beam with a 90 degree beam
    indexes = np.linspace(0, len(beams)-1, len(beams))
    beam = int(indexes[np.abs(beams[:,2] - beam_angle) == 0][0])

    # Read in PFISR data
    (pfisr_time, pfisr_alt,
     e_density, de_density,
     beam_az,
     beam_el) = get_pfisr_parameters(beam, pfisr_file,
                                     low_cutoff = 0, high_cutoff=1e12)

    return (pfisr_time, pfisr_alt, e_density, de_density,
            beam_az, beam_el)
        
def get_pfisr_parameters(beam_num, file, low_cutoff=0,
                         high_cutoff=1e20):
    """ Function to get time, altitude, and electron density for a
    specified beam
    INPUT
    beam_num
        type: int
        about: pfisr beam number
    file
        type: h5py file
        about: file containing all of the pfisr data
    low_cutoff=0
        type: int
        about: values below this will be set to this value
    high_cutoff=1e20
        type: int
        about: values above this will be set to this value
    OUTPUT
    utc_time
        type: datetime array
        about: array of times corresponding to data
    altitude
        type: float array
        about: array of altitudes for pfisr data
    e_density
        type: float array
        about: array of electron density pfisr data
    de_density
        type: float array
        about: array of error in electron density data
    beam_az
        type: float array
        about: array of azimuthal pfisr beam angles
    beam_el
        type: float array
        about: array of elevation pfisr beam angles
    """
    # Convert timestamps to datetime object
    #...use starting time as timestamp
    unix_time = np.array(file['Time']['UnixTime'])[:,0]
    utc_time = np.array([datetime.datetime.utcfromtimestamp(d)
                         for d in unix_time])

    # Get the altitude array
    altitude = np.array(file['NeFromPower']['Altitude'])[beam_num, :]
    #...convert to kilometers
    altitude = altitude/1000

    # Get the uncorrected number density array
    e_density = np.array(file['NeFromPower']
                         ['Ne_NoTr'])[:, beam_num, :]
    #...and error
    de_density = np.array(file['NeFromPower']
                          ['errNe_NoTr'])[:, beam_num, :]
    de_density = np.transpose(de_density)
    #...and filter it assuming data outside of range is bad
    e_density[e_density < low_cutoff] = 0.0
    e_density[e_density > high_cutoff] = 0.0
    #...and take the transpose
    e_density = np.transpose(e_density)

    # Get information about the beam look direction
    beam_az = np.round(np.array(file['BeamCodes'])[:, 1], 1)
    beam_el = np.round(np.array(file['BeamCodes'])[:, 2], 1)

    return (utc_time, altitude, e_density,
            de_density, beam_az, beam_el)

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
