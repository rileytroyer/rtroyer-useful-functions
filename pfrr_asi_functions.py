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
import gc
import h5py
from matplotlib import animation
from matplotlib import colors as mcolors
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
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

    except Exception as e:
        print(e)
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
                filenames = []
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
                          np.shape(image)[1]), dtype=np.uint8)
    
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

        # Turn any nan values into numbers 
        image = np.nan_to_num(image, nan=np.nanmin(image))

        # Recast to 0 to 255 for uint8 datatype
        image = image - np.min(image)
        image[image > 255] = 255

        # Time from filename
        year = int(file[14:18])
        month = int(file[18:20])
        day = int(file[20:22])
        hour = int(file[23:25])
        minute = int(file[25:27])
        second = int(file[27:29])
        time = dt(year, month, day, hour, minute, second)
        
        # Rotate image if necessary
        # The ASI sensor wasn't aligned with N-S-E-W before 2018
        #...and the FOV isn't centered on the sensor, 
        #...so I need to correct for these
        if date.year < 2018:
            x_shift = 7
            y_shift = 12
            fov_radius = 243
            angle = 90
        else:
            x_shift = 7
            y_shift = -5
            fov_radius = 243
            angle = 0
            
        image = ndimage.rotate(image, angle=angle, reshape=False,
			      mode='constant', cval=np.nanmin(image))

	# Recast from 0 to 255 again, I think something in the rotation looses the initial casting
        image = image - np.min(image)
        image[image > 255] = 255

        # Make sure array is uint8 to reduce size
        image = image.astype(np.uint8)

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
        dataset.attrs['wavelength'] = wavelength
        
    # If specified delete all individual files
    if del_files == True:
        shutil.rmtree(img_dir + str(date) + '-' + wavelength + '/')

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

def create_timestamped_movie(date, wavelength='428',
                             img_base_dir = ('../data/pfrr-asi-data/'
                                             'pfrr-images/'),
                             save_base_dir = ('../data/pfrr-asi-data/'
                                              'pfrr-images/movies/')):
    
    """Function to create a movie from PFRR ASI files with a timestamp and frame number.
    Includes a timestamp, and frame number. 
    DEPENDENCIES
        h5py, datetime.datetime, matplotlib.pyplot, matplotlib.animation
        rtroyer_useful_functions
    INPUT
    date
        type: datetime
        about: day to create movie for
    wavelength = '428'
        type: str
        about: which wavelength images are being used
    save_base_dir = ('../figures/themis-figures/')
        type: string
        about: base directory to store keogram image
    img_base_dir = ('../data/themis-asi-data/'
                                           'themis-images/')
        type: string
        about: base directory to where themis asi images are stored.
    OUTPUT
    none
    """
    
    # Select file with images
    img_file = (img_base_dir + '/all-images-'
                + wavelength + '-' + str(date) + '.h5')

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
                           color='white')
    time_str = str(all_times[0])
    time_label = axpic.text(120, 500,
                            time_str,
                            fontweight='bold',
                            color='white')

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
                      + wavelength + '-' + str(date)
                      + '.mp4')
    writer = animation.writers['ffmpeg'](fps=fps)
    anim.save(event_movie_fn,
              writer=writer, dpi=150)

    # Close h5py file
    pfrr_file.close()

    # Reset large image array
    all_images = None
