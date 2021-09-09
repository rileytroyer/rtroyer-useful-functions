#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:08:25 2020

@author: rntroyer
"""

from bs4 import BeautifulSoup
import cdflib
from datetime import datetime as dt
import h5py
import numpy as np
import os
import requests
import rtroyer_useful_functions as rt_func
import shutil
import wget


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

def download_themis_images(date, asi,
                           base_url=('http://themis.ssl.berkeley.edu'
                                     + '/data/themis/thg/l1/asi/'),
                           save_dir = ('../data/themis-asi-data/'
                                       'themis-images/')):
    """Function to download themis images for a specified date.
    DEPENDENCIES
        os, wget
    INPUT
    date
        type: datetime
        about: date to download images from
    asi
        type: string
        about: 4 letter themis station
    base_url
        type: string
        about: online database of images
    OUTPUT
    none
    """
     
    # Select files for day and wavelength
    day_str = (str(date.year)
               + str(date.month).zfill(2)
               + str(date.day).zfill(2))
    asi_url = base_url + asi + '/'
    year_url = asi_url + str(date.year) + '/'
    month_url = year_url + str(date.month).zfill(2) + '/'
    
    # Create a directory to store files if it doesn't exist
    img_dir = (save_dir + asi + '/individual-images/' 
	      + str(date) + '/')
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
        
    # All files for month
    file_urls = get_url_paths(month_url)
    
    # Filter to specific day
    file_urls = [f for f in file_urls if day_str in f]
    
    # Remove full day summary file and then sort
    file_urls = [f for f in file_urls if not day_str + '_' in f]
    file_urls = sorted(file_urls)
    
    # Download each of the files
    for n, file_url in enumerate(file_urls):
        
        # First check if file is already downloaded
        if os.path.exists(img_dir + file_url[67:]):
            continue

        result = None
        counter = 0
        
        while result is None:
            
            # Break out of loop after 10 iterations
            if counter > 10:
                break
                
            try:
                wget.download(file_url, img_dir 
                              + file_url[67:])

                result = True
                
            except:
                pass
            
            counter = counter + 1

def create_image_stack(date, asi,
                       img_base_dir = ('../data/themis-asi-data/'
                       'themis-images/')):
    """Function to create a numpy image stack from THEMIS images
    DEPENDENCIES
        os, cdflib, numpy, datetime.datetime, rtroyer_useful_functions
    INPUT
    date
        type: datetime
        about: day to process image files for
    asi
        type: string
        about: which asi to create stack for
    img_base_dir = ('../data/themis-asi-data/'
                       'themis-images/')
        type: string
        about: base directory to where images are stored
    OUTPUT
    all_images
        type: array
        about: array with all the image data
    all_times
        type: list
        about: list with all of the timestamps for the images
    """
    
    # Directory where images are stored
    img_dir = (img_base_dir + asi + '/individual-images/' 
               + str(date) + '/')

    # All files for day
    files = [f for f in os.listdir(img_dir) if not f.startswith('.')]
    files = sorted(files)

    # Initialize list to store all times in
    all_times = []

    # Loop through all images and store in array
    for n, file in enumerate(files):

        # Read in cdf file
        cdf_file = cdflib.CDF(img_dir + file)

        # Get images
        images = cdf_file.varget('thg_asf_' + asi)

        # Recast each image to 0 to 255 for uint8
        for m, image in enumerate(images):

            # Remove any possible nan values
            image = np.nan_to_num(image, nan=np.nanmin(image))
            
            # Shift values so lowest is zero
            image = image - np.min(image)
            image = image/np.max(image)
            image = image*255

            # Write back to array
            images[m, :, :] = image

        # And time
        times = cdf_file.varget('thg_asf_' + asi + '_epoch')

        # Convert time to datetime
        times = [dt.utcfromtimestamp(cdflib.cdfepoch.unixtime(t)
                                           [0])
                 for t in times]

        # If this is the first file initialize array
        if n==0: 
            all_images = images.astype(np.uint8)

        # Otherwise append it it
        if n > 0:
            all_images = np.append(all_images, images, axis=0)

        # Also append to all times list
        all_times.extend(times)
        
        # Update progress
        rt_func.update_progress((n+1)/len(files))
        
    return all_images, all_times

def store_images_hdf5(date, all_images, all_times, asi,
                      del_files=False,
                      save_dir = ('../data/themis-asi-data/'
                                  'themis-images/'),
                      img_base_dir = ('../data/themis-asi-data/'
                                      'themis-images/')):
    """Function to store an array of images to an HDF5 file.
    DEPENDENCIES
        h5py, shutil
    INPUT
    date
        type: datetime
        about: day to process image files for
    all_images
        type: array
        about: array with all images
    all_times
        type: array
        about: array with all times
    asi
        type: str
        about: 4 letter themis asi camera
    del_file = False
        type: bool
        about: whether to delete individual files after saving h5 file
    save_dir = '../data/pfrr-asi-data/pfrr-images/'
        type: str
        about: where to save the images, program will create
               separate directories for each day within this.
    img_dir = '../data/pfrr-asi-data/pfrr-images/individual-images/'
        type: str
        about: where the individual images are stored.
               This is used to delete files if specified.
    OUTPUT
    none
    """

    # Convert time to integer so it can be stored in h5
    timestamps = [int(t.timestamp()) for t in all_times]

    # Create a new HDF5 file if it doesn't already exist
    filename = 'all-images-' + asi + '-' + str(date) + '.h5'
        
    with h5py.File(save_dir + asi + '/' + filename, "w") as hdf5_file:

        # Create a dataset in the file for the images
        dataset = hdf5_file.create_dataset('images',
                                           np.shape(all_images),
                                           dtype='uint8',
                                           data=all_images)
        
        # Also dataset for the times
        dataset_time = hdf5_file.create_dataset('timestamps',
                                                np.shape(timestamps),
                                                dtype='uint64',
                                                data=timestamps)
        
        # Include data attributes as well
        dataset_time.attrs['about'] = ('UT POSIX Timestamp. '
                                       'Use datetime.fromtimestamp '
                                       'to convert.')
        dataset.attrs['asi'] = asi
        
    # If specified delete all individual files
    if del_files == True:
        shutil.rmtree(img_base_dir + asi + '/individual-images/' 
                      + str(date) + '/')