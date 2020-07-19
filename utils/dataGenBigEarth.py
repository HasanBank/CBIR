

import os
import csv
import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import lmdb
import pyarrow as pa
import torch
from skimage.transform import resize
import pdb

def interp_band(bands, img10_shape=[120,120]):
    """ 
    https://github.com/lanha/DSen2/blob/master/utils/patches.py
    """
    bands_interp = np.zeros([bands.shape[0]] + img10_shape).astype(np.float32)
    
    for i in range(bands.shape[0]):
        bands_interp[i] = resize(bands[i] / 30000, img10_shape, mode='reflect') * 30000

    return bands_interp


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)



class dataGenBigEarthLMDB:

    def __init__(self, bigEarthPthLMDB=None, imgTransform=None, state='train', upsampling=False, 
                train_csv=None, val_csv=None, test_csv=None, isSentinel2 = False):

        self.env = lmdb.open(bigEarthPthLMDB, readonly=True, lock=False, readahead=False, meminit=False)
        self.imgTransform = imgTransform
        self.train_bigEarth_csv = train_csv
        self.val_bigEarth_csv = val_csv
        self.test_bigEarth_csv = test_csv
        self.state = state
        self.upsampling = upsampling
        self.patch_names = []
        self.readingCSV()
        self.isSentinel2 = isSentinel2


    def readingCSV(self):
        if self.state == 'train':
            with open(self.train_bigEarth_csv, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    self.patch_names.append(row[0])

        elif self.state == 'val':
            with open(self.val_bigEarth_csv, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    self.patch_names.append(row[0])
        else:
            with open(self.test_bigEarth_csv, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    self.patch_names.append(row[0])

    def __len__(self):

        return len(self.patch_names)

    def __getitem__(self, idx):
       
        patch_name = self.patch_names[idx]
        if self.isSentinel2:
            patch_name = self.s1NameToS2(patch_name)
            

        if not self.upsampling:
            return self._getData(patch_name, idx)
        else:
            return self._getDataUp(patch_name, idx)

    def _getData(self, patch_name, idx):
        
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode())

        if not self.isSentinel2:
            polarVH, polarVV, multiHots = loads_pyarrow(byteflow)
            sample = {'patchName': patch_name , 'polarVH':polarVH.astype(np.float32), 'polarVV':polarVV.astype(np.float32) , 'label': multiHots.astype(np.float32)}

        else:  
            bands10, bands20, bands60, multiHots = loads_pyarrow(byteflow)
            sample = {'patchName': patch_name , 'bands10':bands10.astype(np.float32), 'bands20':bands20.astype(np.float32), 'bands60':bands60.astype(np.float32), 'label': multiHots.astype(np.float32)}


        if self.imgTransform is not None:
            sample = self.imgTransform(sample)
        
        return sample

    def _getDataUp(self, patch_name, idx):
        

        
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode())

        bands10, bands20, bands60, multiHots = loads_pyarrow(byteflow)

        bands20 = interp_band(bands20)
        bands60 = interp_band(bands60)

        sample = {'patchName':patch_name ,'bands10':bands10.astype(np.float32), 'bands20':bands20.astype(np.float32), 'bands60':bands60.astype(np.float32), 'label': multiHots.astype(np.float32)}

        if self.imgTransform is not None:
            sample = self.imgTransform(sample)
        
        return sample

    def s1NameToS2(self,s1Name):        
        s2Name = s1Name.replace('S1_','')
        return s2Name


class ConcatDataset(object):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)



class Normalize(object):
    def __init__(self, channels_mean, channels_std,isSentinel2):
        
        self.isSentinel2 = isSentinel2
        
        if not self.isSentinel2:
            self.polarVH_mean = channels_mean['polarVH_mean']
            self.polarVH_std = channels_std['polarVH_std']
            
            self.polarVV_mean = channels_mean['polarVV_mean']
            self.polarVV_std = channels_std['polarVV_std']
        else:
            self.bands10_mean = channels_mean['bands10_mean']
            self.bands10_std = channels_std['bands10_std']
    
            self.bands20_mean = channels_mean['bands20_mean']
            self.bands20_std = channels_std['bands20_std']
            
            self.bands60_mean = channels_mean['bands60_mean']
            self.bands60_std = channels_std['bands60_std']

    def __call__(self, sample):
        
        if not self.isSentinel2:
            polarVH, polarVV, label, patchName = sample['polarVH'], sample['polarVV'], sample['label'], sample['patchName']
         
            for t, m, s in zip(polarVH, self.polarVH_mean, self.polarVH_std):
                t.sub_(m).div_(s)
            
            for t, m, s in zip(polarVV, self.polarVV_mean, self.polarVV_std):
                t.sub_(m).div_(s)
            
            return {'polarVH':polarVH, 'polarVV':polarVV, 'label':label, 'patchName': patchName }


        
        else:
            band10, band20, band60, label, patchName = sample['bands10'], sample['bands20'], sample['bands60'], sample['label'], sample['patchName']
            
            for t, m, s in zip(band10, self.bands10_mean, self.bands10_std):
                t.sub_(m).div_(s)
            
            for t, m, s in zip(band20, self.bands20_mean, self.bands20_std):
                t.sub_(m).div_(s)
            
            for t, m, s in zip(band60, self.bands60_mean, self.bands60_std):
                t.sub_(m).div_(s)
        
            return {'bands10':band10, 'bands20':band20, 'bands60':band60, 'label':label, 'patchName': patchName }


    


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self,isSentinel2):
        self.isSentinel2 = isSentinel2
    
    def __call__(self, sample):
        
        if not self.isSentinel2:
            polarVH, polarVV, label, patchName = sample['polarVH'], sample['polarVV'], sample['label'], sample['patchName']
            sample = {'polarVH':torch.tensor(polarVH), 'polarVV':torch.tensor(polarVV),'label':label, 'patchName':patchName}

        else:
            band10, band20, band60, label, patchName = sample['bands10'], sample['bands20'], sample['bands60'],sample['label'], sample['patchName']
            sample = {'bands10':torch.tensor(band10), 'bands20':torch.tensor(band20), 'bands60':torch.tensor(band60), 'label':label, 'patchName':patchName}

        return sample





