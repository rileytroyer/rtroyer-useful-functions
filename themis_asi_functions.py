#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:08:25 2020

@author: rntroyer
"""

import cdflib
from datetime import datetime as dt
import dask
import dask.array as da
import gc
import glob
import h5py
from matplotlib import animation
from matplotlib import colors as mcolors
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
import numpy as np
import os
import rtroyer_useful_functions as rt_func
from scipy import ndimage
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
    if not os.path.exists(save_dir + asi + '/'):
        os.mkdir(save_dir + asi + '/')
    if not os.path.exists(save_dir + asi + '/individual-images/'):
        os.mkdir(save_dir + asi + '/individual-images/')
        
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

def themis_asi_to_hdf5(date, asi, del_files = True,
                       save_base_dir = ('../data/themis-asi-data/themis-images/'),
                       img_base_dir = ('../data/themis-asi-data/themis-images/')
                      ):
    """Function to convert themis asi images
    to 8-bit grayscale images and then write them to an h5 file.
    INPUT
    date
        type: datetime
        about: date to perform image conversion and storage for
    asi
        type: string
        about: which THEMIS camera to use
    del_files = True
        type: bool
        about: whether to delete the individual files after program runs
    save_base_dir = ('../data/themis-asi-data/themis-images')
        type: string
        about: base directory to save the images to
    img_base_dir = ('../data/themis-asi-data/themis-images/')
        type: string
        about: base directory where the individual images are stored
    OUTPUT
    none
    """    
    def read_cdf(filename):

        """Function to use cdflib to read in cdf file and 
        output a numpy array
        INPUT
        filename
            type: string
            about: cdf file to be read in
        OUTPUT
        img
            type: numpy array
            about: image data array
        """

        # Read in cdf file
        cdf_file = cdflib.CDF(filename)

        # Get images
        img = cdf_file.varget('thg_asf_' + asi)

        # Close file
        cdf_file.close()

        return img

    def get_cdf_times(filename):
        """Function to use cdflib to get times from cdf file and
        output a list
        INPUT
            filename
            type: string
            about: cdf file to be read in
        OUTPUT
        times
            type: list
            about: list of times associated with images
        """

        # Read in cdf file
        cdf_file = cdflib.CDF(filename)    

        # Get epoch times
        times = cdf_file.varget('thg_asf_' + asi + '_epoch')

        # Convert time to datetime
        times = [dt.utcfromtimestamp(cdflib.cdfepoch.unixtime(t)
                                           [0])
                 for t in times]

        # Close file
        cdf_file.close()   

        return times

    def get_cdf_dims(filename):
        """Function to get the data dimensions from a CDF file
        INPUT
            filename
            type: string
            about: cdf file to be read in
        OUTPUT
        times
            type: list
            about: list with shape of data
        """

        # Read in cdf file
        cdf_file = cdflib.CDF(filename)

        # Get image dimensions
        info = cdf_file.varinq('thg_asf_' + asi)

        # Construct image shape
        img_shape = (info['Last_Rec']+1, info['Dim_Sizes'][0], info['Dim_Sizes'][1])

        # Close file
        cdf_file.close()

        return img_shape

    def process_img(img, time):

        """Function to process THEMIS ASI image. Plots in log scale and output
        to 8-bit
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

#         # First rotate the image as needed
#         if time.year < 2018:
#             angle = -90
#         else:
#             angle = 0
        # Rotate image to north is at top
        angle = 180
        img = ndimage.rotate(img, angle=angle, reshape=False,
                             mode='constant', cval=np.nanmin(img),
                             axes=(2, 1))
        
        # Flip image so east is on left, to match looking up at sky
        img = np.flip(img, axis=2)

        # Make smallest pixel value zero
        img = abs(img - np.min(img))

        # Set a maximum pixel value
        max_pixel_val = 2**16

        # Set anything larger to this value
        img[img>max_pixel_val] = max_pixel_val

        # Logarithmically scale image
        img = (255/np.sqrt(1 + max_pixel_val)) * np.sqrt(1 + img)

        # Clip to 0 to 255
        img[img>255] = 255

        # Convert to uint8
        img = img.astype('uint8')


        return img


    # Change directories to specific asi
    img_base_dir = img_base_dir + asi + '/individual-images/'
    save_base_dir = save_base_dir + asi + '/'
    
    # Check if directory to save to exits, if not create
    if not os.path.exists(save_base_dir):
        os.mkdir(save_base_dir)

    output = []

    # Directory where images are stored
    img_dir = (img_base_dir + str(date) + '/')

    # All files for day
    files = [f for f in os.listdir(img_dir)]
    files = sorted(files)

    # Initialize list to store all times in
    all_times = []

    # Loop through all files and construct full list of times
    for file in files:

        # Append file times to master list
        all_times.extend(get_cdf_times(img_dir + file))

    # Convert times to array
    all_times = np.array(all_times)

    # Convert to integer timestamp for easier storage in h5 file
    timestamps = np.array([int(t.timestamp()) for t in all_times])

    # Read all images into dask array
    filepathnames = glob.glob(img_dir + '*.cdf')

    # Read files into dask array
    #...this is a little tricky since not all the image stacks are the same dimension
    darray = [dask.delayed(read_cdf)(fn) for fn in filepathnames]
    darray = [da.from_delayed(darray[x],
                              shape=get_cdf_dims(filepathnames[x]),
                              dtype='uint16')
              for x in np.arange(0, len(darray))]
    darray = da.concatenate(darray, axis=0)

    # Write images to h5 dataset
    h5file = save_base_dir + 'all-images-' + str(date) + '-' + asi + '.h5'

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
        img_ds.attrs['wavelength'] = 'white'

        # Loop through 1000 images at a time
        img_chunk = 1000
        for n_img, img in enumerate(darray[0::img_chunk]):

            # Read all 100 images into a numpy array
            img = np.array(darray[n_img*img_chunk:
                                  n_img*img_chunk + img_chunk])

            # Process the image
            img = process_img(img, dt.fromtimestamp(timestamps[0]))

            # Write image to dataset
            img_ds[n_img*img_chunk:
                   n_img*img_chunk+img_chunk:, :, :] = img

            # Update how far along code is
            rt_func.update_progress((n_img+1)/int(darray.shape[0]/img_chunk))

    # If specified to delete files, remove individual images
    if del_files == True:
        shutil.rmtree(img_dir)

    return output

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
                + str(date) + '-' + asi + '.h5')

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
    if not os.path.exists(save_base_dir + asi + '/movies/'):
        os.mkdir(save_base_dir + asi + '/movies/')
        
    event_movie_fn = (save_base_dir + asi + '/movies/' 
                      + str(date) + '-' + asi
                      + '.mp4')
    writer = animation.writers['ffmpeg'](fps=fps)
    anim.save(event_movie_fn,
              writer=writer, dpi=150)

    # Close h5py file
    themis_file.close()

    # Reset large image array
    all_images = None
