#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:08:25 2020

@author: rntroyer
"""

import cdflib
from datetime import datetime as dt
import gc
import h5py
from matplotlib import animation
from matplotlib import colors as mcolors
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
import numpy as np
import os
import rtroyer_useful_functions as rt_func
import shutil
import wget


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
    file_urls = rt_func.get_url_paths(month_url)
    
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

        # Make super bright values not nearly as bright
        images[images>10000] = 10000
    
        # Get the dimest and brighest pixels
        min_pixel = np.min(images)
        max_pixel = np.max(images)

        # Recast each image to 0 to 255 for uint8
        for m, image in enumerate(images):

            # Remove any possible nan values
            image = np.nan_to_num(image, nan=np.nanmin(image))
            
            # Shift values so lowest is zero
            image = image - min_pixel
            
            # Make super bright values not nearly as bright
            image[image>10000] = 10000
            
            # Convert to 0 to 255
            image = image/max_pixel
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

def create_themis_keogram(date, asi,
                          save_fig = False, close_fig = True,
                          save_base_dir = ('../figures/themis-figures/'),
                          img_base_dir = ('../data/themis-asi-data/'
                                           'themis-images/')):
    """Function to create a keogram image for THEMIS camera.
    DEPENDENCIES
        h5py, datetime.datetime, numpy, matplotlib.pyplot, 
        matplotlib.colors, matplotlib.dates, gc
    INPUT
    date
        type: datetime
        about: date to download images from
    asi
        type: string
        about: 4 letter themis station
    save_fig = False
        type: bool
        about: whether to save the figure or not
    close_fig = True
        type: bool
        about: whether to close the figure, useful when producing 
                many plots
    """
    
    # Select file with images
    img_file = (img_base_dir + asi + '/all-images-'
                + asi + '-' + str(date) + '.h5')

    themis_file = h5py.File(img_file, "r")

    # Get times from file
    all_times = [dt.fromtimestamp(d) for d in themis_file['timestamps']]

    # Get all the images too
    all_images = themis_file['images']

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
        save_dir = (save_base_dir + asi + '-keograms/')
        plt.savefig(save_dir + asi + '-' + str(date) + '.jpg', 
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

def create_timestamped_movie(date, asi,
                 save_base_dir = '../data/themis-asi-data/themis-images/',
                 img_base_dir = '../data/themis-asi-data/themis-images/'
                ):
    
    """Function to create a movie from THEMIS ASI files with a timestamp and frame number.
    Includes a timestamp, and frame number. 
    DEPENDENCIES
        h5py, datetime.datetime, matplotlib.pyplot, matplotlib.animation
    INPUT
    date
        type: datetime
        about: day to process image files for
    asi
        type: str
        about: 4 letter themis asi location
    OUTPUT
    none
    """
    
    # Select file with images
    img_file = (img_base_dir + asi + '/all-images-'
                + asi + '-' + str(date) + '.h5')

    themis_file = h5py.File(img_file, "r")

    # Get times from file
    all_times = [dt.fromtimestamp(d) for d in themis_file['timestamps']]

    # Get all the images too
    all_images = themis_file['images']

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
    frame_num = axpic.text(10, 250, '00000', fontweight='bold',
                           color='red')
    time_str = str(all_times[0])
    time_label = axpic.text(120, 250,
                            time_str,
                            fontweight='bold',
                            color='red')

    plt.tight_layout()

    def updatefig(frame):
        """Function to update the animation"""

        # Set new image data
        img.set_data(all_images[frame])
        # And the frame number
        frame_num.set_text(str(frame).zfill(5))
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
    event_movie_fn = (save_base_dir + asi + '/movies/' 
                      + asi + '-' + str(date)
                      + '.mp4')
    writer = animation.writers['ffmpeg'](fps=fps)
    anim.save(event_movie_fn,
              writer=writer, dpi=150)

    # Close h5py file
    themis_file.close()

    # Reset large image array
    all_images = None
