#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:08:25 2020

@author: rntroyer
"""

from astropy.io import fits
import datetime
from datetime import datetime as dt
import ftplib
import h5py
from multiprocessing import Pool
import numpy as np
import os
import rtroyer_useful_functions as rt_func
from scipy import ndimage
import shutil
import smtplib
import ssl
import wget


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


def download_pfrr_images(date,
                         base_url = 
                         'ftp://optics.gi.alaska.edu/PKR/DASC/RAW/',
                         wavelength = '428',
                         save_dir = ('../data/pfrr-asi-data/pfrr-images/'
                                     'individual-images/')):
    """Function to download image files from the
    Poker Flat Research Range (PFRR) all-sky imager images.
    DEPENDENCIES
        multiprocessing.Pool, os
    INPUT
    date
        type: datetime
        about: day to download files for
    base_url
        type: url string
        about: base url for ftp images
    wavelength = '428'
        type: str
        about: which wavelength images are being used
    save_dir = '../data/pfrr-asi-data/pfrr-images/individual-images/'
        type: str
        about: where to save the images, program will create
               separate directories for each day within this.
    OUTPUT
    none
    """

    # Select files for day and wavelength
    file_urls = get_pfrr_asi_filenames(date)
    
    # Filter to wavelength if available
    if date.year > 2009:
        file_urls = [f for f in file_urls if '_0' 
                     + wavelength + '_' in f]

    # Create a directory to store files
    day_dir = (save_dir + str(date) + '-' + wavelength + '/')
    if not os.path.exists(day_dir):
        os.mkdir(day_dir)

    # Create a string to input into job, include url and dir
    job_input = [day_dir + '###' + f for f in file_urls]
    
    # Keep trying to download if all files aren't downloaded
    finished = False
    while finished == False:
        
        # Use download function in threads to download all files
        #...not sure how many processes are possible, but 25 
        #...seems to work, 50 does not.
        pool = Pool(processes=25)
        pool.map(job, job_input)
        
        # Terminate threads when finished
        pool.terminate()
        pool.join()

        # As a last step check to make sure all files were downloaded
        finished_check = []
        for file_url in file_urls:
            finished_check.append(os.path.exists(day_dir 
                                                 + file_url[54:]))
        if False not in finished_check:
            finished = True

def create_image_stack(date, wavelength = '428',
                       save_dir = ('../data/pfrr-asi-data/pfrr-images/'
                                   'individual-images/')):
    """Function to create a numpy image stack from PFRR images
    DEPENDENCIES
        astropy.io.fits, os, numpy, rtroyer_useful_functions,
        scipy.ndimage
    INPUT
    date
        type: datetime
        about: day to process image files for
    wavelength = '428'
        type: str
        about: which wavelength images are being used
    save_dir = '../data/pfrr-asi-data/pfrr-images/individual-images/'
        type: str
        about: where to save the images, program will create
               separate directories for each day within this.
    OUTPUT
    all_images
        type: array
        about: array with all images
    all_times
        type: array
        about: array with all times
    """
    
    # Directory where images are stored
    img_dir = (save_dir + str(date) + '-' + wavelength + '/')

    # All files for day
    files = [f for f in os.listdir(img_dir) if not f.startswith('.')]
    files = sorted(files)

    old_image = 0
    
    # Initialize array to store values in on first image
    image_file = fits.open(img_dir + files[0])
    image = image_file[0].data
    image_file.close()
    all_images = np.zeros((len(files), 
                          np.shape(image)[0],
                          np.shape(image)[1]), dtype=np.uint16)
    
    # Initialize time as list to make datatype easier
    all_times = []

    # Loop through all images and store in dictionary
    for n, file in enumerate(files):

        # Read in image file
        #...encountered an issue where file isn't read sometimes
        try:
            image_file = fits.open(img_dir + file)
            image = image_file[0].data
            image_file.close()
                
        except:
            
            # If this is the first file exit program
            if n==0:
                message = ('Subject: Program Issue! \n'
                           + 'The first file of ' + str(date)
                           + ' ' + file + ' could not be read...'
                           + 'ending program.')
                rt_func.send_email(message)
                exit()
                
            # Otherwise just set image to previous image
            image = old_image

        # Time from filename
        year = int(file[14:18])
        month = int(file[18:20])
        day = int(file[20:22])
        hour = int(file[23:25])
        minute = int(file[25:27])
        second = int(file[27:29])
        time = dt(year, month, day, hour, minute, second)

        # Process image
        image = np.nan_to_num(image, nan=np.nanmin(image))
        
        # Convert to uint16 to reduce size
        image = image.astype(np.uint16)
        
        # Rotate image if necessary
        # The ASI sensor wasn't aligned with N-S-E-W before 2018
        #...and the FOV isn't centered on the sensor, 
        #...so I need to correct for these
        if date.year < 2018:
            x_shift = 7
            y_shift = 12
            fov_radius = 243
            angle = 75
        else:
            x_shift = 7
            y_shift = -5
            fov_radius = 243
            angle = 0
            
        image = ndimage.rotate(image, angle=angle, reshape=False)

        # Store in array
        all_images[n, :, :] = image
        all_times.append(time)
        
        # Update old image
        old_image = image
        
        # Update progress
        rt_func.update_progress((n+1)/len(files))
    
    return all_images, all_times


def store_images_hdf5(date, all_images, all_times, wavelength='428',
                      del_files=False,
                      save_dir = ('../data/pfrr-asi-data/'
                                  'pfrr-images/'),
                      img_dir = ('../data/pfrr-asi-data/pfrr-images/'
                                 'individual-images/')):
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
    wavelength = '428'
        type: str
        about: which wavelength images are being used
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
    filename = 'all-images-' + wavelength + '-' + str(date) + '.h5'
        
    with h5py.File(save_dir + filename, "w") as hdf5_file:

        # Create a dataset in the file for the images
        dataset = hdf5_file.create_dataset('images',
                                           np.shape(all_images),
                                           dtype='uint16',
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
        dataset.attrs['wavelength'] = wavelength
        
    # If specified delete all individual files
    if del_files == True:
        shutil.rmtree(img_dir + str(date) + '-' + wavelength + '/')