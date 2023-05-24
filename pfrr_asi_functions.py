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
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = clahe.apply(image)

    # Scale back to 0 to 255 values and 8-bit
    image = cv2.convertScaleAbs(image, alpha=(255.0/np.max(image)))

    return image

def pfrr_asi_to_hdf5_8bit_clahe(date:datetime.datetime, save_base_dir:str, img_base_dir:str,
                                wavelength:str='558', del_files:bool = False, processes:int=1):
    """Function to convert 428, 558, 630 nm PFRR images for an entire
    night to an 8-bit grayscale image and then write them to an h5 file.
    INPUT
    date - date to perform image conversion and storage for
    save_base_dir - about: base directory to save the images to
    img_base_dir - about: base directory where the individual images are stored
    wavelength - which wavelength to use. White combines all three.
               Options: 428, 558, 630
    del_files - whether to delete the individual files after program runs
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

            # Close and join pools, I think not doing this may cause shared memory issues in /dev/shm
            pool.close()
            pool.join()

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

def create_timestamped_movie(date, img_base_dir, save_base_dir, wavelength='558'):
    
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
