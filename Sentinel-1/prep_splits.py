#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script creates splits with LMDB files from Sentinel-1 
# image patches based on csv files that contain patch names.
# 
# This is based on: https://gitlab.tubit.tu-berlin.de/rsim/bigearthnet-models/blob/master/prep_splits.py
# Modified for Sentinel-1 Images

# prep_splits.py --help can be used to learn how to use this script.
#
# 
# Author: Hasan Bank
# Email: hasan.bank@campus.tu-berlin.de
# Date: 08 Oct 2020
# Version: 1.0.2
# Usage: prep_splits.py [-h] [-r ROOT_FOLDER] [-s1 S1_ROOT_FOLDER] [-o OUT_FOLDER]
#                       [-n PATCH_NAMES [PATCH_NAMES ...]] [-name NAME_LMDB]

from __future__ import print_function
import argparse
import os
import csv
from pytorch_utils import prep_lmdb_files

GDAL_EXISTED = False
RASTERIO_EXISTED = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        'This script creates LMDB files for the Sentinel-1 train, validation and test splits')
    parser.add_argument('-r', '--root_folder', dest = 'root_folder',
                        help = 'root folder path contains multiple patch folders')
    parser.add_argument('-s1', '--s1_root_folder', dest = 's1_root_folder',
                        help = 'sentinel 1 root folder path contains multiple patch folders')
    parser.add_argument('-o', '--out_folder', dest = 'out_folder', 
                        help = 'folder path containing resulting LMDB files')
    parser.add_argument('-n', '--splits', dest = 'splits', help = 
                        'csv files each of which contain list of patch names', nargs = '+')
    parser.add_argument('-name', type=str, dest='name', help='name of the creating lmdb file')


    args = parser.parse_args()

    # Checks the existence of patch folders and populate the list of patch folder paths
    folder_path_list = []
    if args.root_folder:
        if not os.path.exists(args.root_folder):
            print('ERROR: folder', args.root_folder, 'does not exist')
            exit()
    else:
        print('ERROR: folder', args.patch_folder, 'does not exist')
        exit()

    # Checks the existence of required python packages
    try:
        import gdal
        GDAL_EXISTED = True
        print('INFO: GDAL package will be used to read GeoTIFF files')
    except ImportError:
        try:
            import rasterio
            RASTERIO_EXISTED = True
            print('INFO: rasterio package will be used to read GeoTIFF files')
        except ImportError:
            print('ERROR: please install either GDAL or rasterio package to read GeoTIFF files')
            exit()
    
    try:
        import numpy as np
    except ImportError:
        print('ERROR: please install numpy package')
        exit()

    if args.splits:
        try:
            patch_names_list = []
            split_names = []
            for csv_file in args.splits:
                patch_names_list.append([])
                split_names.append(os.path.basename(csv_file).split('.')[0])
                with open(csv_file, 'r') as fp:
                    csv_reader = csv.reader(fp, delimiter=',')
                    for row in csv_reader:
                        patch_names_list[-1].append(row[0].strip())    
        except:
            print('ERROR: some csv files either do not exist or have been corrupted')
            exit()


        
    prep_lmdb_files(
            args.root_folder, 
            args.s1_root_folder,
            args.out_folder,
            patch_names_list,
            GDAL_EXISTED,
            RASTERIO_EXISTED,
            args.name
        )