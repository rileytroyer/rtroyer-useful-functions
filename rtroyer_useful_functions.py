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


def create_keogram(img_stack, median=False, n=5):
    """Function to create a keogram array from a set of images.
    INPUT
    img_stack
        type: 3D array 
        description: A stack of all the images to create the keogram from
    n=5
        type: integer
        description: If taking an average this is the averaging range
    median=False
        type: boolean
        description: Whether to take the median of a few columns
    OUTPUT
    keogram
        type: 2D array 
        description: Array representing the keogram
    """
    
    # Create an array to store the keogram data
    keogram = np.zeros([img_stack.shape[1], img_stack.shape[0]])

    # Find the middle of picture
    middle = int(img_stack.shape[2]/2)

    # Then slice the array to get the keogram
    if median==False:
        keogram = img_stack[:, :, middle]
    if median==True:
        keogram = np.median(img_stack[:, :, (middle-n):(middle+n)],
                          axis=2)
    #...finally rotate by 90 degrees
    keogram = np.rot90(keogram)
        
            
    return keogram


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

def normalize_image(image, uint=True):
    """
    Function to turn images into a 0-256 scale image array
    input and output are both numpy arrays
    if uint=True datatype of output array is uint8
    """
    
    # First look for minimum value, this will be black
    black = np.min(image)
    #...then shift values so black is 0
    image_normalized = image-black
    
    # Set everything above 255 equal to 255, this is white
    image_normalized[image_normalized > 255] = 255

    if uint==True:
        return image_normalized.astype('uint8')
    else:
        return image_normalized

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

def send_email(message):
    """Function to send an email notification.
    INPUT
    message
        type: str
        about: Should be in form of Subject: subject \n
               message.
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

def get_pfrr_asi_filenames(date):

    """Function to get file pathnames for pfrr asi
    INPUT
    date
        type: datetime
        about: date to get pathnames for
    OUTPUT
    filenames
        type: list
        about: list of all file pathnames
    """
    
    # Login to the ftp as public
    ftp_link = 'optics.gi.alaska.edu'
    ftp = ftplib.FTP(ftp_link)
    ftp.login()
    #...access the imager directory (DASC - digital all sky camera)
    rel_imager_dir = ('/PKR/DASC/RAW/' 
                      + str(date.year).zfill(4) + '/'
                      + str(date.year).zfill(4) + str(date.month).zfill(2)
                      + str(date.day).zfill(2) + '/')


    # Find which years there is data for
    #...try until it works, the server often returns an error
    try:
         # Set current working directory to the ftp directory
        ftp.cwd(rel_imager_dir)
        #...store directories in dictionary
        filenames = ['ftp://' + ftp_link + rel_imager_dir 
                     + f for f in ftp.nlst()]

    except:
        result = None
        counter = 0
        while result is None:

            # Break out of loop after 10 iterations
            if counter > 10:
                print('Unable to get data from: ftp://' + ftp_link 
                      + rel_imager_dir)
                break

            try:
                # Set current working directory to the ftp directory
                ftp = ftplib.FTP(ftp_link)
                ftp.login()
                ftp.cwd(rel_imager_dir)
                #...store directories in dictionary
                filenames = ['ftp://' + ftp_link + rel_imager_dir 
                             + f + '/' for f in ftp.nlst()]
                result = True

            except:
                pass

            counter = counter + 1
            
    return filenames

def job(job_input):
    """Function to pass to thread process to download file
    INPUT
    job_input
        type: string
        about: string that contains the local directory to store
                the downloaded files and the file url to download
                these are seperated by the ### characters
    OUTPUT
    none"""
    day_dir, file_url = job_input.split('###')
    
    # Check if file is already downloaded
    if os.path.exists(day_dir + file_url[54:]):
        return
    
    result = None
    counter = 0
    while result is None:
        # Break out of loop after 10 iterations
        if counter > 10:
            break
        try:
            wget.download(file_url, day_dir 
                          + file_url[54:])

            result = True
        except:
            pass
        counter = counter + 1
