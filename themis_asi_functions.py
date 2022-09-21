#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:08:25 2020

@author: rntroyer
"""

from datetime import datetime
import gc
import dask
import dask.array as da
import h5py
import logging
import math
from matplotlib import animation
from matplotlib import pyplot
from moviepy.editor import *
from multiprocessing import Pool
import numpy
import os
import re
from scipy.io import readsav
import shutil
import subprocess
import sys
import themis_imager_readfile


def download_themis_images(date, asi, save_dir = '../../../asi-data/themis/'):
    """Function to download raw .pgm.gz files from stream0 of THEMIS
    ASIs. Data is downloaded from https://data.phys.ucalgary.ca/
    INPUT
    date
        type: datetime (make sure it is full datetime not datetime.date)
        about: which date to download images for
    asi
        type: str
        about: 4 letter code for station to download images for
    save_dir = '../../../asi-data/themis/'
        type: str
        about: path to directory to save images
    OUTPUT
    logging. I recommend writing to file by running this at the start of the code:
    
    logging.basicConfig(filename='themis-script.log',
                    encoding='utf-8',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
    """
    
    logging.info('Starting download script for {} and {}.'.format(asi, date.date()))
    
    date_string = (str(date.year).zfill(4) + '/' 
                   + str(date.month).zfill(2) + '/'
                   + str(date.day).zfill(2) + '/')

    # URL for entire THEMIS ASI project
    themis_url = 'data.phys.ucalgary.ca/data/sort_by_project/THEMIS/asi/'

    # URL for skymap
    skymap_url = themis_url + 'skymaps/' + asi + '/'

    # URL for images
    img_url = themis_url + 'stream0/' + date_string + asi + '*/'

    # Get the matched skymap for the date

    # Get output of sync to find the directories
    try:
        skymap_dirs = subprocess.check_output(['rsync', 
                                               'rsync://' + skymap_url]).splitlines()
        skymap_dirs = [str(d.split(b' ')[-1], 'UTF-8') for d in skymap_dirs[1:]]

    except Exception as e:
        logging.critical('Unable to access skymap server: {}. '
                         'Server may be down. Stopping.'.format(skymap_url))
        logging.critical('Exception: {}'.format(e))
        raise

    # Convert to datetimes
    skymap_dates = [d.split('_')[1] for d in skymap_dirs]
    skymap_dates = [datetime.strptime(d, '%Y%m%d') for d in skymap_dates]

    # Find time difference from each skymap date
    time_diffs = numpy.array([(date - d).total_seconds() for d in skymap_dates])

    # Find the closest map
    skymap_dir = skymap_dirs[numpy.where(time_diffs > 0,
                                         time_diffs, numpy.inf).argmin()]

    skymap_url = skymap_url + skymap_dir + '/'

    # Create directories to store data

    # Does directory exist for imager?
    save_asi_dir = save_dir + asi + '/'
    if not os.path.exists(save_asi_dir):
        os.makedirs(save_asi_dir)

    # Does a temporary directory for raw images and skymap files exist?
    tmp_dir = save_asi_dir + 'tmp/'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    tmp_img_dir = tmp_dir + str(date.date()) + '/'
    if not os.path.exists(tmp_img_dir):
        os.mkdir(tmp_img_dir)
        download_imgs = True
    else:
        logging.info('Images already exists for {},'
                     ' so will not download.'.format(date.date()))
        download_imgs = False

    # Do the images need to be downloaded?
    if download_imgs == True:
        
        # Download skymap
        logging.info('Downloading skymap from {}...'.format(skymap_url))
        try:
            subprocess.run(['rsync', '-vzrt', 'rsync://' + skymap_url + '*.sav',
                             tmp_img_dir], stdout=subprocess.DEVNULL)
            logging.info('Successfully downloaded skymap.'
                             ' It is saved at {}.'.format(tmp_img_dir))
        except Exception as e:
            logging.critical('Unable to download skymap:{}. Stopping.'.format(skymap_url))
            logging.critical('Exception: {}'.format(e))
            raise

        # Download images
        logging.info('Downloading images from {}...'.format(img_url))
        try:
            subprocess.run(['rsync', '-vzrt', 'rsync://' + img_url,
                             tmp_img_dir], stdout=subprocess.DEVNULL)
            logging.info('Successfully downloaded images.'
                         ' They are saved at {}.'.format(tmp_img_dir))
        except Exception as e:
            logging.critical('Unable to download images:{}. Stopping.'.format(img_url))
            logging.critical('Exception: {}'.format(e))
            raise
            
    logging.info('Finished download script for {} and {}.'.format(asi, date.date()))

def themis_asi_to_hdf5(date, asi, del_files = False,
                       save_dir = ('../../../asi-data/themis/')):
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
    save_dir = ('../../../asi-data/themis/')
        type: string
        about: base directory to save the images to
    OUTPUT
    logging. I recommend writing to file by running this at the start of the code:
    
    logging.basicConfig(filename='themis-script.log',
                    encoding='utf-8',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
    """     
    
    def process_img(img):

        """Function to process THEMIS ASI image. Plots in log scale and output
        to 8-bit
        INPUT
        img
            type: array
            about: image data in array
        OUTPUT
        img
            type: array
            about: 8-bit processed image
        """

        # Make smallest pixel value zero
        img = abs(img - numpy.min(img))

        # Set a maximum pixel value
        max_pixel_val = 2**16

        # Set anything larger to this value
        img[img>max_pixel_val] = max_pixel_val
        
        # Scale image
        img = (255/numpy.sqrt(1 + max_pixel_val)) * numpy.sqrt(1 + img)

        # Clip to 0 to 255
        img[img>255] = 255

        # Convert to uint8
        img = img.astype('uint8')


        return img

    def read_img(filename):

        """Function to read in THEMIS stream0 image file and 
        output a numpy array
        INPUT
        filename
            type: string
            about: img file to be read in
        OUTPUT
        img
            type: numpy array
            about: image data array
        """

        # Read in img file
        img_file = themis_imager_readfile.read(filename)

        # Get images
        img = img_file[0]

        return img    
    
    # Write images to h5 dataset
    logging.info('Starting h5 file creation script for {} and {}...'.format(asi,
                                                                            date.date()))

    h5file = save_dir + asi + '/all-images-' + str(date.date()) + '-' + asi + '.h5'
    
    # Directory with images
    tmp_img_dir = save_dir + asi + '/tmp/' + str(date.date()) + '/'
    
    if not os.path.exists(tmp_img_dir):
        logging.critical('Images are not downloaded. Try running download_themis_images.')
    
    # Read in skymap
    skymap_file = [f for f in os.listdir(tmp_img_dir) if f.endswith('.sav')][0]

    try:
        # Try reading IDL save file
        skymap = readsav(tmp_img_dir + skymap_file, python_dict=True)['skymap']

        # Get arrays
        skymap_alt = skymap['FULL_MAP_ALTITUDE'][0]
        skymap_glat = skymap['FULL_MAP_LATITUDE'][0][:, 0:-1, 0:-1]
        skymap_glon = skymap['FULL_MAP_LONGITUDE'][0][:, 0:-1, 0:-1]
        skymap_elev = skymap['FULL_ELEVATION'][0]
        skymap_azim = skymap['FULL_AZIMUTH'][0]
        
        logging.info('Read in skymap file from: {}'.format(skymap_file))
        
    except Exception as e:
        logging.error('Unable to read skymap file: {}.'
                         ' Creating file without it.'.format(tmp_img_dir + skymap_file))
        logging.error('Exception: {}'.format(e))
        
        skymap_alt = numpy.array(['Unavailable'])
        skymap_glat = numpy.array(['Unavailable'])
        skymap_glon = numpy.array(['Unavailable'])
        skymap_elev = numpy.array(['Unavailable'])
        skymap_azim = numpy.array(['Unavailable'])

    # Does the downloaded image directory exists?
    if not os.path.exists(tmp_img_dir):
        logging.critical('Images do not exist at {}'.format(tmp_img_dir))

    hour_dirs = os.listdir(tmp_img_dir)
    hour_dirs = sorted([d for d in hour_dirs if d.startswith('ut')])

    # Construct a list of pathnames to each file for day
    filepathnames = []

    for hour_dir in hour_dirs:

        # Name of all images in hour
        img_files = sorted(os.listdir(tmp_img_dir + hour_dir))
        img_files = [tmp_img_dir + hour_dir + '/' + f for f in img_files]

        # Add to master list
        filepathnames.append(img_files)

    with h5py.File(h5file, 'w') as h5f:

        # Initialize the datasets for images and timestamps
        img_ds = h5f.create_dataset('images', shape=(256, 256, 0),
                                    maxshape=(256, 256, None),
                                    dtype='uint8')

        time_ds = h5f.create_dataset('timestamps', shape=(0,),
                                     maxshape=(None,),
                                     dtype='uint64')

        alt_ds = h5f.create_dataset('skymap_alt', shape=skymap_alt.shape,
                                     dtype='float', data=skymap_alt)        
        
        glat_ds = h5f.create_dataset('skymap_glat', shape=skymap_glat.shape,
                                     dtype='float', data=skymap_glat)

        glon_ds = h5f.create_dataset('skymap_glon', shape=skymap_glon.shape,
                                     dtype='float', data=skymap_glon)

        elev_ds = h5f.create_dataset('skymap_elev', shape=skymap_elev.shape,
                                     dtype='float', data=skymap_elev)

        azim_ds = h5f.create_dataset('skymap_azim', shape=skymap_azim.shape,
                                     dtype='float', data=skymap_azim)

        # Loop through each hour, process and write images to file
        logging.info('Processing and writing images to file...')

        try:
            for hour_filepathnames in filepathnames:

                # Read the data files
                img, meta, problematic_files = themis_imager_readfile.read(hour_filepathnames,
                                                                           workers=2)

                # Extract datetimes from file
                datetimes = [datetime.strptime(m['Image request start'],
                                                 '%Y-%m-%d %H:%M:%S.%f %Z') for m in meta]

                # Convert times to integer format
                timestamps = numpy.array([int(t.timestamp()) for t in datetimes])

                # Process the image
                img = process_img(img)

                # Write image to dataset. This requires resizing
                img_ds.resize(img_ds.shape[2] + img.shape[2], axis=2)
                img_ds[:, :, -img.shape[2]:] = img

                # Write timestamp to dataset
                time_ds.resize(time_ds.shape[0] + timestamps.shape[0], axis=0)
                time_ds[-timestamps.shape[0]:] = timestamps

        except Exception as e:
            logging.critical('Unable to write images to file. Stopping.'
                             ' Deleting h5 file and, if specified, images.')
            logging.critical('Exception: {}'.format(e))
            
            # Delete h5 file
            os.remove(h5file)
            
            # Delete the raw image files if specified
            if del_files == True:
                logging.info('Deleting directory: {}'.format(tmp_img_dir))
                shutil.rmtree(tmp_img_dir)
            raise

        # Add attributes to datasets
        time_ds.attrs['about'] = ('UT POSIX Timestamp.'
                                  ' Use datetime.fromtimestamp '
                                  'to convert. Time is start of image.'
                                  ' 1 second exposure.')
        img_ds.attrs['wavelength'] = 'white'
        img_ds.attrs['station_latitude'] = float(meta[0]['Geodetic latitude'])
        img_ds.attrs['station_longitude'] = float(meta[0]['Geodetic Longitude'])
        alt_ds.attrs['about'] = 'Altitudes for different skymaps.'
        glat_ds.attrs['about'] = 'Geographic latitude at pixel corner, excluding last.'
        glon_ds.attrs['about'] = 'Geographic longitude at pixel corner, excluding last.'
        elev_ds.attrs['about'] = 'Elevation angle of pixel center.'
        azim_ds.attrs['about'] = 'Azimuthal angle of pixel center.'
        
    # Delete the raw image files if specified
    if del_files == True:
        logging.info('Deleting directory: {}'.format(tmp_img_dir))
        shutil.rmtree(tmp_img_dir)

    logging.info('Finished h5 file creation script for {} and {}.'
                 ' File is saved to: {}'.format(asi, date.date(), h5file))
    
# def create_themis_keogram(date, asi,
#                           save_fig = False, close_fig = True,
#                           save_base_dir = ('../figures/themis-figures/'),
#                           img_base_dir = ('../data/themis-asi-data/'
#                                            'themis-images/')):
#     """Function to create a keogram image for THEMIS camera.
#     DEPENDENCIES
#         h5py, datetime.datetime, numpy, matplotlib.pyplot, 
#         matplotlib.colors, matplotlib.dates, gc
#     INPUT
#     date
#         type: datetime
#         about: date to download images from
#     asi
#         type: string
#         about: 4 letter themis station
#     save_fig = False
#         type: bool
#         about: whether to save the figure or not
#     close_fig = True
#         type: bool
#         about: whether to close the figure, useful when producing 
#                 many plots
#     """
    
#     # Select file with images
#     img_file = (img_base_dir + asi + '/all-images-'
#                 + asi + '-' + str(date) + '.h5')

#     themis_file = h5py.File(img_file, "r")

#     # Get times from file
#     all_times = [dt.fromtimestamp(d) for d in themis_file['timestamps']]

#     # Get all the images too
#     all_images = themis_file['images']

#     # Get keogram slices
#     keogram_img = all_images[:, :, int(all_images.shape[2]/2)]
    
#     # Do some minor processing to keogram
    
#     # Replace nans with smallest value
#     keogram_img = np.nan_to_num(keogram_img,
#                                 nan=np.nanmin(keogram_img))
    
#     # Rotate by 90 degrees counterclockwise
#     keogram_img = np.rot90(keogram_img)
    
#     # Finally flip, so north is at top in mesh plot
#     keogram_img = np.flip(keogram_img, axis=0)

#     # Construct time and altitude angle meshes to plot against
#     altitudes = np.linspace(0, 180, keogram_img.shape[0])
#     time_mesh, alt_mesh = np.meshgrid(all_times, altitudes)

#     # Setup figure
#     fig, ax = plt.subplots()

#     # Plot keogram as colormesh
#     ax.pcolormesh(time_mesh, alt_mesh, keogram_img,
#                   norm=mcolors.LogNorm(),
#                   cmap='gray', shading='auto')

#     # Axis and labels
#     ax.set_title('keogram of ' + str(date),
#                  fontweight='bold', fontsize=14)
#     ax.set_ylabel('Azimuth Angle (S-N)',
#                   fontweight='bold', fontsize=14)
#     ax.set_xlabel('UT1 Time (HH:MM)',
#                   fontweight='bold', fontsize=14)
#     fig.autofmt_xdate()
#     h_fmt = mdates.DateFormatter('%H:%M')
#     ax.xaxis.set_major_formatter(h_fmt)
#     ax.tick_params(axis='x', which='major', labelsize=14)
#     ax.tick_params(axis='y', which='major', labelsize=14)

#     plt.tight_layout()

#     # Save the figure if specified
#     if save_fig == True:
#         save_dir = (save_base_dir + asi + '-keograms/')
#         plt.savefig(save_dir + asi + '-' + str(date) + '.jpg', 
#                     dpi=250)

#     # Close the figure and all associated
#     #...not sure if I need to clear all of these
#     #...but figure it doesn't hurt
#     if close_fig == True:
#         #...axis
#         plt.cla()
#         #...figure
#         plt.clf()
#         #...figure windows
#         plt.close('all')
#         #...clear memory
#         gc.collect()

def movie_job(job_input):
    """Function to create timestamped movie from input images and times. 
    Outputs to .mp4 file with specified filename.
    INPUT
    job_input
        type: list of lists
        about: [filepathname.mp4, list of datetimes, array of images]
    """
    # Get times from input
    all_times = job_input[1]

    # Get all the images too
    all_images = job_input[2]

    # CREATE MOVIE
    img_num = all_images.shape[2]
    fps = 20.0


    # Construct an animation
    # Setup the figure
    fig, axpic = pyplot.subplots(1, 1)

    # No axis for images
    axpic.axis('off')

    # Plot the image
    img = axpic.imshow(numpy.flipud(all_images[:, :, 0]),
                       cmap='gray', animated=True)

    # Add frame number and timestamp to video
    frame_num = axpic.text(10, 250, '00000', fontweight='bold',
                           color='red')
    time_str = str(all_times[0])
    time_label = axpic.text(120, 250,
                            time_str,
                            fontweight='bold',
                            color='red')

    pyplot.tight_layout()

    def updatefig(frame):
        """Function to update the animation"""

        # Set new image data
        img.set_data(numpy.flipud(all_images[:, :, frame]))
        # And the frame number
        start_frame = int(job_input[0].split('/')[-1].split('.')[0][11:])
        frame_num.set_text(str(frame + start_frame).zfill(5))
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
    pyplot.close(fig)


    # Use ffmpeg writer to save animation
    event_movie_fn = (job_input[0])
    writer = animation.writers['ffmpeg'](fps=fps)
    anim.save(event_movie_fn,
              writer=writer, dpi=150)

def create_timestamped_movie(date, asi, workers=1,
                 save_dir = '../../../asi-data/themis/',
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
    workers=1
        type: int
        about: how many processes to create movies with.
                number can match number of available cpu cores.
    save_base_dir = '../../../asi-data/themis/'
        type: string
        about: where to save the movie files to
    img_base_dir = '../../../asi-data/themis/'
        type: string
        about: where the image files are stored
    OUTPUT
    none
    """

    # Select file with images
    logging.info('Starting timestamped movie script for {} and {}.'.format(asi,
                                                                           date.date()))

    img_file = (save_dir + asi + '/all-images-'
                + str(date.date()) + '-' + asi + '.h5')

    logging.info('Reading in h5 file: {}'.format(img_file))

    try:
        # Read in h5 file
        themis_file = h5py.File(img_file, "r")

        # Get times from file
        all_times = [datetime.fromtimestamp(d) for d in themis_file['timestamps']]

        # Get all the images
        all_images = themis_file['images']

    except Exception as e:
        logging.critical('There was an issue reading in the h5 file. Stopping.')
        logging.critical('Exception: {}'.format(e))
        raise

    # Check if directory to store movies exists
    movie_dir = save_dir + asi + '/movies/'
    if not os.path.exists(movie_dir):
        os.mkdir(movie_dir)

    # Check if directory to store temporary frames exists
    if not os.path.exists(save_dir + 'tmp-frames/'):
        os.mkdir(save_dir + 'tmp-frames/')


    # Split images and times into smaller portions
    # this allows us to speed up the process with parallel computing

    # How many smaller movies to make, these will get combined into 1 at the end
    bins = 10
    chunk_size = math.ceil(len(all_times)/bins)

    # Define a list to be able to input into the parallelization job
    job_input = []

    # Loop through each chunk and set as one job input
    for n in range(0, len(all_times), chunk_size):

        # Filename for movie chunk
        filename = save_dir + 'tmp-frames/tmp-frames-' + str(n) + '.mp4'

        # Append to job input, need filename, times, and images
        job_input.append([filename, all_times[n:n+chunk_size],
                          all_images[:, :, n:n+chunk_size]])

    # Delete first image array to save ram as this is pretty big
    del all_images, all_times
    gc.collect()

    # Start multiprocessing
    # be aware this can use a fairly large amount of RAM.
    # I often see around 5GB used.
    logging.info('Starting {} movie creating processes.'
                 ' Tmp movies will be combined into one at the end.'.format(workers))

    try:
        pool = Pool(processes=workers)
        pool.map(movie_job, job_input)

        # Terminate threads when finished
        pool.terminate()
        pool.join()

    except Exception as e:
        logging.critical('There was an issue creating the tmp movie files. Stopping.')
        logging.critical('Exception: {}'.format(e))
        raise

    logging.info('Finished creating tmp movies.')

    # List of all tmp movies
    tmp_movie_files = [f for f in os.listdir(save_dir + 'tmp-frames/')
                       if f.startswith('tmp') & f.endswith('.mp4')]

    # Make sure files are sorted properly
    def num_sort(string):
        return list(map(int, re.findall(r'\d+', string)))[0]

    tmp_movie_files.sort(key=num_sort)

    # Add in path
    tmp_movie_files = [save_dir + 'tmp-frames/' + f for f in tmp_movie_files]

    # File to write
    full_movie_pathname = movie_dir + 'full-movie-' + str(date.date()) + '-' + asi + '.mp4'

    # Concatenate smaller tmp movies into a full one
    logging.info('Combining tmp movies into one file at: {}.'.format(full_movie_pathname))
    try:
        clips = []

        for filename in tmp_movie_files:
            clips.append(VideoFileClip(filename))

        video = concatenate_videoclips(clips, method='chain')
        video.write_videofile(full_movie_pathname)

    except Exception as e:
        logging.warning('There was an issue creating the full movie file. Stopping.')
        logging.warning('Exception: {}'.format(e))

    logging.info('Full movie file created. Deleting tmp files.')

    try:
        # Remove all tmp movie files
        for file in tmp_movie_files:
            os.remove(file)

    except Exception as e:
        logging.warning('Could not delete tmp movie files.')
        logging.warning('Exception: {}'.format(e))

    logging.info('Finished timestamped movie script for {} and {}.'.format(asi,
                                                                           date.date()))