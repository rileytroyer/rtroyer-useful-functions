#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:08:25 2020

@author: rntroyer
"""

from bs4 import BeautifulSoup
import os
import requests
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
    img_dir = (save_dir + asi + '/' + str(date) + '/')
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