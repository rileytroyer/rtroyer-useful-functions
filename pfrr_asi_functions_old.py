#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:08:25 2020

@author: rntroyer
"""

from astropy.io import fits
import datetime
from datetime import datetime as dt
from dask.array.image import imread as dask_imread
import ftplib
import gc
import h5py
import logging
from matplotlib import animation
from matplotlib import colors as mcolors
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
import multiprocessing
import numpy as np
import os
from scipy import ndimage
import shutil
import smtplib
import ssl
import wget
import cv2

output=[]

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

    except Exception as e:
        logging.warning(f'Could not get image filepaths. Stopped with error {e}')
        result = None
        counter = 0
        while result is None:

            # Break out of loop after 10 iterations
            if counter > 10:
                logging.error(f'Unable to get data from: ftp://{ftp_link}{rel_imager_dir}, skipping.')
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

def download_job(job_input):
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
            logging.error(f'Unable to download image from: {file_url}, skipping.')
            break
        try:
            wget.download(file_url, day_dir 
                          + file_url[54:], bar=None)

            result = True
        except:
            pass
        counter = counter + 1


def download_pfrr_images(date, save_dir, wavelength = '558', processes=25):
    """Function to download image files from the
    Poker Flat Research Range (PFRR) all-sky imager images.
    DEPENDENCIES
        multiprocessing.Pool, os
    INPUT
    date
        type: datetime
        about: day to download files for
    save_dir
        type: str. example: '../data/pfrr-asi-data/pfrr-images/individual-images/'
        about: where to save the images, program will create
               separate directories for each day within this.
    wavelength = '428'
        type: str
        about: which wavelength images are being used
    processes=25
        type: int
        about: how many multiprocessing process to download images with
    OUTPUT
    none
    """

    logging.info(f'Starting to download files for {date} and {wavelength}.')

    # Select files for day and wavelength
    file_urls = get_pfrr_asi_filenames(date)
    
    # Filter to wavelength if available
    if date.year > 2009:
        file_urls = [f for f in file_urls if '_0' 
                     + wavelength + '_' in f]

    # Create a directory to store files
    day_dir = (save_dir + str(date) + '-' + wavelength + '/')
    logging.info(f'Downloading images to {day_dir}.')
    if not os.path.exists(day_dir):
        os.makedirs(day_dir)

    # Create a string to input into job, include url and dir
    download_job_input = [day_dir + '###' + f for f in file_urls]
    
    # Keep trying to download if all files aren't downloaded
    finished = False
    while finished == False:
        
        # Use download function in threads to download all files
        #...not sure how many processes are possible, but 25 
        #...seems to work, 50 does not on my computer.
        with multiprocessing.Pool(processes=processes) as pool:
            pool.map(download_job, download_job_input)

        # As a last step check to make sure all files were downloaded
        finished_check = []
        for file_url in file_urls:
            finished_check.append(os.path.exists(day_dir 
                                                 + file_url[54:]))
        if False not in finished_check:
            finished = True

    logging.info('Finished downloading images.')

def read_process_img_clahe(filename):
    """Function to use astropy.io.fits to read in fits file,
    process it with a CLAHE method and 
    output a numpy array. Note for CLAHE input array needs to be unsigned int
    INPUT
    filename
        type: string
        about: fits file to be read in
    OUTPUT
    image
        type: numpy array
        about: processed image data array
    """

    # Read in the image
    fits_file = fits.open(filename)
    image = fits_file[0].data.astype('uint16')
    fits_file.close()

    # Image processing
    #clahe = cv2.createCLAHE(clipLimit=10)
    #image = clahe.apply(image)

    return image

def pfrr_asi_to_hdf5_8bit_clahe(date, save_base_dir, img_base_dir,
                                wavelength='558',
                                del_files = False, processes=25):
    """Function to convert 428, 558, 630 nm PFRR images for an entire
    night to an 8-bit grayscale image and then write them to an h5 file.
    INPUT
    date
        type: datetime
        about: date to perform image conversion and storage for
    save_base_dir
        type: string. example: '../data/pfrr-asi-data/pfrr-images/'
        about: base directory to save the images to
    img_base_dir
        type: string. example: '../data/pfrr-asi-data/pfrr-images/individual-images'
        about: base directory where the individual images are stored
    wavelength='white'
        type: string
        about: which wavelength to use. White combines all three.
               Options: 428, 558, 630
    del_files = True
        type: bool
        about: whether to delete the individual files after program runs
    update_progress = True
        type: bool
        about: whether to update progress of creating the h5 file.
    OUTPUT
    none
    """
    
    # Get directory where images are stored
    dir_wavelength = img_base_dir + str(date) + '-' + wavelength + '/'
    
    # Get a list of files
    files_wavelength = sorted(os.listdir(dir_wavelength))
    # Make sure these are only .fits files
    files_wavelength = [f for f in files_wavelength if f.endswith('.FITS')]
    
    # Extract times from filenames
    times_wavelength = np.array([dt(int(f[14:18]), int(f[18:20]), int(f[20:22]),
                            int(f[23:25]), int(f[25:27]), int(f[27:29]))
                            for f in files_wavelength])

    # Get the full filepath
    files_wavelength = [dir_wavelength + f for f in files_wavelength]
    
    # Convert datetime to integer timestamp
    timestamps = np.array([int(t.timestamp()) for t in times_wavelength])
    # And ISO string
    iso_time = np.array([t.isoformat() for t in times_wavelength]).astype('S26')
    
    # Write images to h5 dataset
    h5file = save_base_dir + 'all-images-' + str(date) + '-' + wavelength + '.h5'

    # Read in a sample image to get correct size for dataset
    sample_image = read_process_img_clahe(files_wavelength[0])
    image_data_shape = (len(files_wavelength), sample_image.shape[0], sample_image.shape[1])

    with h5py.File(h5file, 'w') as h5f:

        # Initialize the datasets for images and timestamps
        img_ds = h5f.create_dataset('images', shape=image_data_shape,
                                    dtype='uint8')

        time_ds = h5f.create_dataset('timestamps', shape=timestamps.shape,
                                     dtype='uint64', data=timestamps)
        iso_time_ds = h5f.create_dataset('iso_time_string', shape=iso_time.shape,
                                         dtype='S26', data=iso_time)

        # Add attributes to datasets
        time_ds.attrs['about'] = ('UT POSIX Timestamp.'
                                  'Use datetime.fromtimestamp '
                                  'to convert.')
        iso_time_ds.attrs['about'] = ('ISO string format for UT time.')
        img_ds.attrs['wavelength'] = wavelength

        logging.info(f'Initialized h5 file: {h5file}. Starting to write data.')

        # Loop through 100 images at a time
        chunk_size = 100

        for n_chunk, files in enumerate(files_wavelength[0::chunk_size]):

            files = files_wavelength[n_chunk*chunk_size : n_chunk*chunk_size+chunk_size]
            
            # Read and process files using multiprocessing
            with multiprocessing.Pool(processes=processes) as pool:
                processed_images = pool.map(read_process_img_clahe, files)

            # Stack the images into a numpy array
            processed_images_array = np.stack(processed_images, axis=0)

            # Write image to dataset
            img_ds[n_chunk*chunk_size:
                   n_chunk*chunk_size+chunk_size:, :, :] = processed_images_array

            logging.info(f'Finished writing {n_chunk*chunk_size} of {len(files_wavelength)} images.')
            
        # If specified to delete files, remove individual images
        if del_files == True:
            shutil.rmtree(dir_wavelength)
    
    logging.info(f'Finished writing data to h5 file.')


def pfrr_asi_to_hdf5(date, wavelength='white', del_files = True,
                     update_progress = True,
                               save_base_dir = ('../data/pfrr-asi-data/pfrr-images/'),
                               img_base_dir = ('../data/pfrr-asi-data/pfrr-images/'
                                          'individual-images/')):
    """Function to convert 428, 558, 630 nm PFRR images for an entire
    night to an 8-bit grayscale image and then write them to an h5 file.
    INPUT
    date
        type: datetime
        about: date to perform image conversion and storage for
    wavelength='white'
        type: string
        about: which wavelength to use. White combines all three.
               Options: 428, 558, 630
    del_files = True
        type: bool
        about: whether to delete the individual files after program runs
    update_progress = True
        type: bool
        about: whether to update progress of creating the h5 file.
    save_base_dir = ('../data/pfrr-asi-data/pfrr-images')
        type: string
        about: base directory to save the images to
    img_base_dir = ('../data/pfrr-asi-data/pfrr-images/'
                    'individual-images/')
        type: string
        about: base directory where the individual images are stored
    OUTPUT
    none
    """
    
    def read_fits(filename):
    
        """Function to use astropy.io.fits to read in fits file and 
        output a numpy array
        INPUT
        filename
            type: string
            about: fits file to be read in
        OUTPUT
        img
            type: numpy array
            about: image data array
        """

        fits_file = fits.open(filename)
        img = fits_file[0].data
        fits_file.close()

        return img

    def process_img(img, time):

        """Function to process PFRR ASI image. Filters out bright and dark pixels
        then converts to 8-bit
        INPUT
        img
            type: array
            about: image data in array
        time
            type: datetime
            about: time associated with image
        OUTPUT
        img
            type: array
            about: 8-bit processed image
        """

        # First rotate the image as needed
        if time.year < 2018:
            angle = -90
        else:
            angle = 0

        img = ndimage.rotate(img, angle=angle, reshape=False,
                             mode='constant', cval=np.nanmin(img),
                             axes=(2, 1))
        
        # Make smallest pixel value zero
        img = abs(img - np.min(img))
        
        # Set a maximum pixel value
        max_pixel_val = 2**16
        
        # Set anything larger to this value
        img[img>max_pixel_val] = max_pixel_val
        
        # Logarithmically scale image
        img = (255/np.log(1 + max_pixel_val)) * np.log(1 + img)
        
        # Clip to 0 to 255
        img[img>255] = 255
        
        # Convert to uint8
        img = img.astype('uint8')
        

        return img
    
    def process_img_clahe(img, time):
        # First rotate the image as needed
        if time.year < 2018:
            angle = -90
        else:
            angle = 0

        img = ndimage.rotate(img, angle=angle, reshape=False,
                             mode='constant', cval=np.nanmin(img),
                             axes=(2, 1))
        
        clahe = cv2.createCLAHE(clipLimit=30)
        img_processed = np.zeros(img.shape)
        for n in range(img.shape[0]):
            img_processed[n, :, :] = clahe.apply(img[n, :, :])

        return img_processed
    
    # Combine rgb images to create white wavelength
    if wavelength == 'white':
        
        # Get a list of all files for each wavelength and extract times
        dir_428 = img_base_dir + str(date) + '-428/'
        dir_558 = img_base_dir + str(date) + '-558/'
        dir_630 = img_base_dir + str(date) + '-630/'

        # Get names of files in each directory
        files_428 = sorted(os.listdir(dir_428))
        files_558 = sorted(os.listdir(dir_558))
        files_630 = sorted(os.listdir(dir_630))
        
        # Read all of the images into seperate dask arrays for each wavelength
        filepathnames_428 = dir_428 + '*.FITS'
        darray_428 = dask_imread(filepathnames_428, imread=read_fits)

        filepathnames_558 = dir_558 + '*.FITS'
        darray_558 = dask_imread(filepathnames_558, imread=read_fits)

        filepathnames_630 = dir_630 + '*.FITS'
        darray_630 = dask_imread(filepathnames_630, imread=read_fits)
        
        # Warn if not the same number of files
        if not len(files_428) == len(files_558) == len(files_630):
            output.append('Warning: not the same number of files for each wavelength.')
            
            # If only off by 1 image remove last image from longer list
            img_lengths = np.array([len(files_428),
                                      len(files_558),
                                      len(files_630)])
            
            if (np.max(img_lengths) - np.min(img_lengths)) < 3:
                
                # Fix files
                files_428 = files_428[0:np.min(img_lengths)]
                files_558 = files_558[0:np.min(img_lengths)]
                files_630 = files_630[0:np.min(img_lengths)]
                
                # Also dask arrays
                darray_428 = darray_428[0:np.min(img_lengths)]
                darray_558 = darray_558[0:np.min(img_lengths)]
                darray_630 = darray_630[0:np.min(img_lengths)]
                
                output.append('Number of files only varied by 1, removing last files.')
            else:
                output.append('Number of files was greater than 1, will not create .h5 file.')
                
        # Create a combined wavelength grayscale array
        #...these values are for converting from RGB to grayscale
        darray = darray_428*0.114 + darray_558*0.299 + darray_630*0.587

        # Extract a list of times for each list of files
        times_428 = np.array([dt(int(f[14:18]), int(f[18:20]), int(f[20:22]),
                              int(f[23:25]), int(f[25:27]), int(f[27:29]))
                              for f in files_428])
        times_558 = np.array([dt(int(f[14:18]), int(f[18:20]), int(f[20:22]),
                              int(f[23:25]), int(f[25:27]), int(f[27:29]))
                              for f in files_558])
        times_630 = np.array([dt(int(f[14:18]), int(f[18:20]), int(f[20:22]),
                              int(f[23:25]), int(f[25:27]), int(f[27:29]))
                              for f in files_630])
        
        # Convert datetime to integer timestamp
        timestamps = np.array([int(t.timestamp()) for t in times_558])
    
    # If not white create for specified wavelength
    else:
        # Get directory where images are stored
        dir_wavelength = img_base_dir + str(date) + '-' + wavelength + '/'
        
        # Get a list of files
        files_wavelength = sorted(os.listdir(dir_wavelength))
        
        # Extract times from filenames
        times_wavelength = np.array([dt(int(f[14:18]), int(f[18:20]), int(f[20:22]),
                              int(f[23:25]), int(f[25:27]), int(f[27:29]))
                              for f in files_wavelength])
        
        # Convert datetime to integer timestamp
        timestamps = np.array([int(t.timestamp()) for t in times_wavelength])
        
        # Read all images into dask array
        filepathnames_wavelength = dir_wavelength + '*.FITS'
        darray = dask_imread(filepathnames_wavelength, imread=read_fits)
        
        
    
    # Write images to h5 dataset
    h5file = save_base_dir + 'all-images-' + str(date) + '-' + wavelength + '.h5'

    with h5py.File(h5file, 'w') as h5f:

        # Initialize the datasets for images and timestamps
        img_ds = h5f.create_dataset('images', shape=darray.shape,
                                    dtype='uint8')

        time_ds = h5f.create_dataset('timestamps', shape=timestamps.shape,
                                     dtype='uint64', data=timestamps)

        # Add attributes to datasets
        time_ds.attrs['about'] = ('UT POSIX Timestamp.'
                                  'Use datetime.fromtimestamp '
                                  'to convert.')
        img_ds.attrs['wavelength'] = wavelength

        # Loop through 100 images at a time
        img_chunk = 100
        for n_img, img in enumerate(darray[0::img_chunk]):

            # Read all 100 images into a numpy array
            img = np.array(darray[n_img*img_chunk:
                                  n_img*img_chunk + img_chunk])

            # Process the image
            img = process_img_clahe(img.astype(np.uint16), dt.fromtimestamp(timestamps[0]))

            # Write image to dataset
            img_ds[n_img*img_chunk:
                   n_img*img_chunk+img_chunk:, :, :] = img

            # Update how far along code is
            if update_progress == True:
           #     rt_func.update_progress((n_img+1)/int(darray.shape[0]/img_chunk))
                continue
            
        # If specified to delete files, remove individual images
        if del_files == True:
            if wavelength == 'white':
                shutil.rmtree(dir_428)
                shutil.rmtree(dir_558)
                shutil.rmtree(dir_630)
            else:
                shutil.rmtree(dir_wavelength)
    
    return output

def pfrr_asi_to_hdf5_16bit(date, wavelength='white', del_files = True,
                            update_progress = True,
                            save_base_dir = ('../data/pfrr-asi-data/pfrr-images/'),
                            img_base_dir = ('../data/pfrr-asi-data/pfrr-images/'
                                          'individual-images/')):
    """Function to convert 428, 558, 630 nm PFRR images for an entire
    night to an 16-bit grayscale image and then write them to an h5 file.
    INPUT
    date
        type: datetime
        about: date to perform image conversion and storage for
    wavelength='white'
        type: string
        about: which wavelength to use. White combines all three.
               Options: 428, 558, 630
    del_files = True
        type: bool
        about: whether to delete the individual files after program runs
    update_progress = True
        type: bool
        about: whether to update progress of creating the h5 file.
    save_base_dir = ('../data/pfrr-asi-data/pfrr-images')
        type: string
        about: base directory to save the images to
    img_base_dir = ('../data/pfrr-asi-data/pfrr-images/'
                    'individual-images/')
        type: string
        about: base directory where the individual images are stored
    OUTPUT
    none
    """
    
    def read_fits(filename):
    
        """Function to use astropy.io.fits to read in fits file and 
        output a numpy array
        INPUT
        filename
            type: string
            about: fits file to be read in
        OUTPUT
        img
            type: numpy array
            about: image data array
        """

        fits_file = fits.open(filename)
        img = fits_file[0].data
        fits_file.close()

        return img
    
    output = []
    
    # Combine rgb images to create white wavelength
    if wavelength == 'white':
        
        # Get a list of all files for each wavelength and extract times
        dir_428 = img_base_dir + str(date) + '-428/'
        dir_558 = img_base_dir + str(date) + '-558/'
        dir_630 = img_base_dir + str(date) + '-630/'

        # Get names of files in each directory
        files_428 = sorted(os.listdir(dir_428))
        files_558 = sorted(os.listdir(dir_558))
        files_630 = sorted(os.listdir(dir_630))
        
        # Read all of the images into seperate dask arrays for each wavelength
        filepathnames_428 = dir_428 + '*.FITS'
        darray_428 = dask_imread(filepathnames_428, imread=read_fits)

        filepathnames_558 = dir_558 + '*.FITS'
        darray_558 = dask_imread(filepathnames_558, imread=read_fits)

        filepathnames_630 = dir_630 + '*.FITS'
        darray_630 = dask_imread(filepathnames_630, imread=read_fits)
        
        # Warn if not the same number of files
        if not len(files_428) == len(files_558) == len(files_630):
            output.append('Warning: not the same number of files for each wavelength.')
            
            # If only off by 1 image remove last image from longer list
            img_lengths = np.array([len(files_428),
                                      len(files_558),
                                      len(files_630)])
            
            if (np.max(img_lengths) - np.min(img_lengths)) < 3:
                
                # Fix files
                files_428 = files_428[0:np.min(img_lengths)]
                files_558 = files_558[0:np.min(img_lengths)]
                files_630 = files_630[0:np.min(img_lengths)]
                
                # Also dask arrays
                darray_428 = darray_428[0:np.min(img_lengths)]
                darray_558 = darray_558[0:np.min(img_lengths)]
                darray_630 = darray_630[0:np.min(img_lengths)]
                
                output.append('Number of files only varied by 1, removing last files.')
            else:
                output.append('Number of files was greater than 1, will not create .h5 file.')
                
        # Create a combined wavelength grayscale array
        #...these values are for converting from RGB to grayscale
        darray = darray_428*0.114 + darray_558*0.299 + darray_630*0.587

        # Extract a list of times for each list of files
        times_428 = np.array([dt(int(f[14:18]), int(f[18:20]), int(f[20:22]),
                              int(f[23:25]), int(f[25:27]), int(f[27:29]))
                              for f in files_428])
        times_558 = np.array([dt(int(f[14:18]), int(f[18:20]), int(f[20:22]),
                              int(f[23:25]), int(f[25:27]), int(f[27:29]))
                              for f in files_558])
        times_630 = np.array([dt(int(f[14:18]), int(f[18:20]), int(f[20:22]),
                              int(f[23:25]), int(f[25:27]), int(f[27:29]))
                              for f in files_630])
        
        # Convert datetime to integer timestamp
        timestamps = np.array([int(t.timestamp()) for t in times_558])
    
    # If not white create for specified wavelength
    else:
        # Get directory where images are stored
        dir_wavelength = img_base_dir + str(date) + '-' + wavelength + '/'
        
        # Get a list of files
        files_wavelength = sorted(os.listdir(dir_wavelength))
        
        # Extract times from filenames
        times_wavelength = np.array([dt(int(f[14:18]), int(f[18:20]), int(f[20:22]),
                              int(f[23:25]), int(f[25:27]), int(f[27:29]))
                              for f in files_wavelength])
        
        # Convert datetime to integer timestamp
        timestamps = np.array([int(t.timestamp()) for t in times_wavelength])
        
        # Read all images into dask array
        filepathnames_wavelength = dir_wavelength + '*.FITS'
        darray = dask_imread(filepathnames_wavelength, imread=read_fits)
        
        
    
    # Write images to h5 dataset
    h5file = save_base_dir + 'all-images-' + str(date) + '-' + wavelength + '.h5'

    with h5py.File(h5file, 'w') as h5f:

        # Initialize the datasets for images and timestamps
        img_ds = h5f.create_dataset('images', shape=darray.shape,
                                    dtype='uint16')

        time_ds = h5f.create_dataset('timestamps', shape=timestamps.shape,
                                     dtype='uint64', data=timestamps)

        # Add attributes to datasets
        time_ds.attrs['about'] = ('UT POSIX Timestamp.'
                                  'Use datetime.fromtimestamp '
                                  'to convert.')
        img_ds.attrs['wavelength'] = wavelength

        # Loop through 100 images at a time
        img_chunk = 100
        for n_img, img in enumerate(darray[0::img_chunk]):

            # Read all 100 images into a numpy array
            img = np.array(darray[n_img*img_chunk:
                                  n_img*img_chunk + img_chunk])

            # Write image to dataset
            img_ds[n_img*img_chunk:
                   n_img*img_chunk+img_chunk:, :, :] = img

            # Update how far along code is
            if update_progress == True:
            #    rt_func.update_progress((n_img+1)/int(darray.shape[0]/img_chunk))
                continue
            
        # If specified to delete files, remove individual images
        if del_files == True:
            if wavelength == 'white':
                shutil.rmtree(dir_428)
                shutil.rmtree(dir_558)
                shutil.rmtree(dir_630)
            else:
                shutil.rmtree(dir_wavelength)
    
    return output

def create_pfrr_keogram(date, wavelength = '428',
                        save_fig = False, close_fig = True,
                        save_base_dir = ('../figures/pfrr-figures/'
                                         'pfrr-keograms-general/'),
                        img_base_dir = ('../data/pfrr-asi-data/'
                                        'pfrr-images/')):
    """Function to create a keogram image for PFRR camera.
    INPUT
    DEPENDENCIES
        h5py, datetime.datetime, numpy, matplotlib.pyplot,
        matplotlib.dates, matplotlib.colors, gc
    date
        type: datetime
        about: date to create keogram for
    wavelength
        type: str
        about: which wavelength images are being used
    save_fig = False
        type: bool
        about: whether to save the figure or not
    close_fig = True
        type: bool
        about: whether to close the figure, useful when producing 
                many plots
    save_base_dir = ('../figures/themis-figures/')
        type: string
        about: base directory to store keogram image
    img_base_dir = ('../data/themis-asi-data/'
                                           'themis-images/')
        type: string
        about: base directory to where themis asi images are stored.
    """
    
    # Select file with images
    img_file = (img_base_dir + '/all-images-'
                + wavelength + '-' + str(date) + '.h5')

    pfrr_file = h5py.File(img_file, "r")

    # Get times from file
    all_times = [dt.fromtimestamp(d) for d in pfrr_file['timestamps']]

    # Get all the images too
    all_images = pfrr_file['images']

    # Get keogram slices
    keogram_img = all_images[:, :, int(all_images.shape[2]/2)]
    
    # Do some minor processing to keogram
    
    # Replace nans with smallest value
    keogram_img = np.nan_to_num(keogram_img,
                                nan=np.nanmin(keogram_img))
    
    # Rotate by 90 degrees counterclockwise
    keogram_img = np.rot90(keogram_img)
    
    # Finally flip, so north is at top in mesh plot
    keogram_img = np.flip(keogram_img, axis=0)

    # Boost contrast
    # keogram_img = rt_func.img_clean_n_boost(keogram_img,
    #                                      weight=0, low=0.001, high=100)

    # Construct time and altitude angle meshes to plot against
    altitudes = np.linspace(0, 180, keogram_img.shape[0])
    time_mesh, alt_mesh = np.meshgrid(all_times, altitudes)

    # Setup figure
    fig, ax = plt.subplots()

    # Plot keogram as colormesh
    ax.pcolormesh(time_mesh, alt_mesh, keogram_img,
                  norm=mcolors.LogNorm(),
                  cmap='gray', shading='auto')

    # Axis and labels
    ax.set_title('keogram of ' + str(date),
                 fontweight='bold', fontsize=14)
    ax.set_ylabel('Azimuth Angle (S-N)',
                  fontweight='bold', fontsize=14)
    ax.set_xlabel('UT1 Time (HH:MM)',
                  fontweight='bold', fontsize=14)
    fig.autofmt_xdate()
    h_fmt = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(h_fmt)
    ax.tick_params(axis='x', which='major', labelsize=14)
    ax.tick_params(axis='y', which='major', labelsize=14)

    plt.tight_layout()

    # Save the figure if specified
    if save_fig == True:
        save_dir = save_base_dir
        plt.savefig(save_dir + wavelength + '-' + str(date) + '.jpg', 
                    dpi=250)

    # Close the figure and all associated
    #...not sure if I need to clear all of these
    #...but figure it doesn't hurt
    if close_fig == True:
        #...axis
        plt.cla()
        #...figure
        plt.clf()
        #...figure windows
        plt.close('all')
        #...clear memory
        gc.collect() 

def create_timestamped_movie(date,img_base_dir, save_base_dir, wavelength='558'):
    
    """Function to create a movie from PFRR ASI files with a timestamp and frame number.
    Includes a timestamp, and frame number. 
    DEPENDENCIES
        h5py, datetime.datetime, matplotlib.pyplot, matplotlib.animation
        rtroyer_useful_functions
    INPUT
    date
        type: datetime
        about: day to create movie for
    save_base_dir
        type: string. example: '../figures/themis-figures/'
        about: base directory to store keogram image
    img_base_dir
        type: string. example: '../data/themis-asi-data/themis-images/'
        about: base directory to where themis asi images are stored.
    wavelength = '428'
        type: str
        about: which wavelength images are being used
    OUTPUT
    none
    """
    
    # Select file with images
    img_file = (img_base_dir + '/all-images-'
                + str(date) + '-' + wavelength + '.h5')

    pfrr_file = h5py.File(img_file, "r")

    # Get times from file
    all_times = [dt.fromtimestamp(d) for d in pfrr_file['timestamps']]

    # Get all the images too
    all_images = pfrr_file['images']

    # CREATE MOVIE

    img_num = all_images.shape[0]
    fps = 20.0


    # Construct an animation
    # Setup the figure
    fig, axpic = plt.subplots(1, 1)

    # No axis for images
    axpic.axis('off')

    # Plot the image
    img = axpic.imshow(all_images[0],
                       cmap='gray', animated=True)

    # Add frame number and timestamp to video
    frame_num = axpic.text(10, 500, '0000', fontweight='bold',
                           color='red')
    time_str = str(all_times[0])
    time_label = axpic.text(120, 500,
                            time_str,
                            fontweight='bold',
                            color='red')

    plt.tight_layout()

    def updatefig(frame):
        """Function to update the animation"""

        # Set new image data
        img.set_data(all_images[frame])
        # And the frame number
        frame_num.set_text(str(frame).zfill(4))
        #...and time
        time_str = str(all_times[frame])
        time_label.set_text(time_str)

        return [img, frame_num, time_label]

    # Construct the animation
    anim = animation.FuncAnimation(fig, updatefig,
                                   frames=img_num,
                                   interval=int(1000.0/fps),
                                   blit=True)

    # Close the figure
    plt.close(fig)


    # Use ffmpeg writer to save animation
    event_movie_fn = (save_base_dir 
                      + str(date) + '-' + wavelength
                      + '.mp4')
    writer = animation.writers['ffmpeg'](fps=fps)
    anim.save(event_movie_fn,
              writer=writer, dpi=150)

    # Close h5py file
    pfrr_file.close()

    # Reset large image array
    all_images = None
