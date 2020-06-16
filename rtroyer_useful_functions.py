#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:08:25 2020

@author: rntroyer
"""

import numpy as np
import datetime
from skimage import exposure
from skimage.restoration import denoise_nl_means, estimate_sigma
from IPython.display import clear_output


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


def update_progress(progress):
    """Function to display a bar with the progress of an action.
    INPUT
    progress
        type: float
        description: action step/total steps of action
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
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block 
                                             + "-" * (bar_length - block),
                                             progress * 100)
    print(text)