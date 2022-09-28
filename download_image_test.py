from themis_asi_functions import *
# datetime object - https://docs.python.org/3/library/datetime.html
from datetime import datetime
import os
import logging
import multiprocessing
# multiprocessing.set_start_method("forkserver")

if __name__ == '__main__':
    logging.basicConfig(filename='themis-script.log',
                        # encoding='utf-8',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info('test code start')

    # current directory
    crtdir = os.getcwd()
    skymap_dir = crtdir + '/skymap/'
    crtdir += '/image/'

    # note from Riley - 2016-10-13, gako, mcgr
    mydate = datetime(2018, 3, 10, 0, 0, 0)
    asi1, asi2 = "pina", "mcgr"

    # # download themis image at crtdir 
    # # good to use
    # print("download images started")
    # download_themis_images(mydate, asi1, save_dir=crtdir)
    # print("download image ended")

    # convert downloaded to hdf5 # skymap? - fixed
    print("h5 convertion started")
    themis_asi_to_hdf5(mydate, asi1, del_files=False, save_dir = crtdir, workers=8)
    print("h5 convertion ended")

    # creat timestamped movie
    print("timestamped movie generating")
    create_timestamped_movie(mydate, asi1, save_dir=crtdir, workers=8)
    print("timestamped movie generated")

