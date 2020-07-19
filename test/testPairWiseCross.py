# -*- coding: utf-8 -*-
#
# This script can be used to train any deep learning model on the BigEarthNet.
#
# To run the code, you need to provide a json file for configurations of the training.
#
# Author: Jian Kang, https://www.rsim.tu-berlin.de/menue/team/dring_jian_kang/
# Email: jian.kang@tu-berlin.de

import os
import random
import math
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import shutil


import argparse


parser = argparse.ArgumentParser(description='PyTorch multi-label classification for testing pairwise')
    
parser.add_argument('--S1LMDBPth', metavar='DATA_DIR',
                        help='path to the saved sentinel 1 LMDB dataset')
parser.add_argument('--S2LMDBPth', metavar='DATA_DIR',
                        help='path to the saved sentinel 2 LMDB dataset')
parser.add_argument('-b', '--batch-size', default=200, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--bits', type=int, default=16, help='number of bits to use in hashing')
parser.add_argument('--checkpoint_pth', '-c', help='path to the pretrained weights file', default=None, type=str)
parser.add_argument('--num_workers', default=8, type=int, metavar='N',
                        help='num_workers for data loading in pytorch')

parser.add_argument('--train_csvS1', metavar='CSV_PTH',
                        help='path to the csv file of train patches')
parser.add_argument('--val_csvS1', metavar='CSV_PTH',
                        help='path to the csv file of val patches')
parser.add_argument('--test_csvS1', metavar='CSV_PTH',
                        help='path to the csv file of test patches')
parser.add_argument('--dataset', metavar='DATA_DIR', help='path to trained data')
parser.add_argument('--k', type=int, default=20, help='number of retrived images per query')
parser.add_argument('--big1000', dest='big1000', action='store_true', help='for small dataset')
parser.add_argument('--serbia', dest='serbia', action='store_true',
                    help='use the serbia patches')
    
    
arguments = parser.parse_args()


import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn


import sys
sys.path.append('../')

from utils.dataGenBigEarth import dataGenBigEarthLMDB, ToTensor, Normalize, ConcatDataset
from utils.metrics import get_mAP, get_k_hamming_neighbours, Recall_score, F1_score, F2_score, Hamming_loss, Subset_accuracy, \
    Accuracy_score, One_error, Coverage_error, Ranking_loss, LabelAvgPrec_score, calssification_report

from utils.ResNet import ResNet50_S1, ResNet50_S2

ORG_LABELS = [
    'Continuous urban fabric',
    'Discontinuous urban fabric',
    'Industrial or commercial units',
    'Road and rail networks and associated land',
    'Port areas',
    'Airports',
    'Mineral extraction sites',
    'Dump sites',
    'Construction sites',
    'Green urban areas',
    'Sport and leisure facilities',
    'Non-irrigated arable land',
    'Permanently irrigated land',
    'Rice fields',
    'Vineyards',
    'Fruit trees and berry plantations',
    'Olive groves',
    'Pastures',
    'Annual crops associated with permanent crops',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland',
    'Moors and heathland',
    'Sclerophyllous vegetation',
    'Transitional woodland/shrub',
    'Beaches, dunes, sands',
    'Bare rock',
    'Sparsely vegetated areas',
    'Burnt areas',
    'Inland marshes',
    'Peatbogs',
    'Salt marshes',
    'Salines',
    'Intertidal flats',
    'Water courses',
    'Water bodies',
    'Coastal lagoons',
    'Estuaries',
    'Sea and ocean'
]

BigEarthNet19_LABELS = [
    'Urban fabric',
    'Industrial or commercial units',
    'Arable land',
    'Permanent crops',
    'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland and sparsely vegetated areas',
    'Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland/shrub',
    'Beaches, dunes, sands',
    'Inland wetlands',
    'Coastal wetlands',
    'Inland waters',
    'Marine waters'
]


def main():
    

    torch.backends.cudnn.enabled = True
    cudnn.benchmark = True

    if arguments.serbia:
        if arguments.big1000:
            #Sentinel-1 Serbia Small Set Statistics
            polars_mean = {
                    'polarVH_mean': [ -15.841419 ],
                    'polarVV_mean': [ -9.30693]
                }

            polars_std = {
                    'polarVH_std': [ 0.77528906 ],
                    'polarVV_std': [ 1.7330942]
            }
            #Sentinel-2 Serbia Small Set Statistics
            bands_mean = {
                        'bands10_mean': [ 444.81293 ,  661.15625, 651.9382, 2561.2258 ],
                        'bands20_mean': [ 1049.933, 2040.9866, 2402.629, 2614.475, 1999.8666, 1312.8848 ],
                        'bands60_mean': [ 323.30414, 2598.8064 ],
                    }
    
            bands_std = {
                        'bands10_std': [ 287.76907, 275.67145, 274.22473, 288.70212 ],
                        'bands20_std': [ 259.82077, 261.15186, 277.17377, 275.40338, 212.71072, 186.5235 ],
                        'bands60_std': [ 248.35155, 279.67813 ]
                }
        else:
            #Sentinel 2 in Serbia Statistics
            bands_mean = {
                            'bands10_mean': [ 458.93423 ,  676.8278,  665.719, 2590.4482],
                            'bands20_mean': [ 1065.233, 2068.3826, 2435.3057, 2647.92, 2010.1838, 1318.5911],
                            'bands60_mean': [ 341.05457, 2630.7898 ],
                        }
        
            bands_std = {
                            'bands10_std': [ 315.86624,  305.07462,  302.11145, 310.93375],
                            'bands20_std': [ 288.43314, 287.29364, 299.83383, 295.51282, 211.81876,  193.92213],
                            'bands60_std': [ 267.79263, 292.94092 ]
                    }
    else:
        if arguments.big1000 :
            
             bands_mean = {'bands10_mean': [ 416.19177 ,  599.8206,  572.7137, 2227.16],
                        'bands20_mean': [ 937.8009, 1796.316, 2085.186, 2279.0896, 1606.1233, 1016.82526],
                        'bands60_mean': [ 330.60297, 2257.2668 ],
                        }
             bands_std = {
                        'bands10_std': [ 300.9185, 292.4436, 319.7186, 437.5927],
                        'bands20_std': [ 311.32214, 364.91702, 402.91406, 413.4657, 249.48933, 238.40024],
                        'bands60_std': [ 260.72678, 364.29974 ]
                        }
        else:
                bands_mean = {
                        'bands10_mean': [ 429.9430203 ,  614.21682446,  590.23569706, 2218.94553375],
                        'bands20_mean': [ 950.68368468, 1792.46290469, 2075.46795189, 2266.46036911, 1594.42694882, 1009.32729131],
                        'bands60_mean': [ 340.76769064, 2246.0605464 ],
                    }
    
                bands_std = {
                        'bands10_std': [ 572.41639287,  582.87945694,  675.88746967, 1365.45589904],
                        'bands20_std': [ 729.89827633, 1096.01480586, 1273.45393088, 1356.13789355, 1079.19066363,  818.86747235],
                        'bands60_std': [ 554.81258967, 1302.3292881 ]
                    }



    modelS1 = ResNet50_S1(arguments.bits)
    modelS2 = ResNet50_S2(arguments.bits)
    
    test_dataGenS1 = dataGenBigEarthLMDB(
                    bigEarthPthLMDB=arguments.S1LMDBPth,
                    isSentinel2 = False,
                    state='test',
                    imgTransform=transforms.Compose([
                        ToTensor(False),
                        Normalize(polars_mean, polars_std,False)
                    ]),
                    upsampling=False,
                    train_csv=arguments.train_csvS1,
                    val_csv=arguments.val_csvS1,
                    test_csv=arguments.test_csvS1
    )
    
    
    
    
    test_dataGenS2 = dataGenBigEarthLMDB(
                    bigEarthPthLMDB=arguments.S2LMDBPth,
                    isSentinel2 = True,
                    state='test',
                    imgTransform=transforms.Compose([
                        ToTensor(True),
                        Normalize(bands_mean, bands_std,True)
                    ]),
                    upsampling=True,
                    train_csv=arguments.train_csvS1,
                    val_csv=arguments.val_csvS1,
                    test_csv=arguments.test_csvS1
    )
    

    
    test_data_loader = DataLoader(ConcatDataset(test_dataGenS1,test_dataGenS2), batch_size=arguments.batch_size, num_workers=arguments.num_workers, shuffle=False, pin_memory=True)

    if torch.cuda.is_available():
        modelS1.cuda()
        modelS2.cuda()
        gpuDisabled = False
    else:
        modelS1.cpu()
        modelS2.cpu()
        gpuDisabled = True

    checkpointPath = arguments.checkpoint_pth
    checkpoint = torch.load(checkpointPath)
    
    modelS1.load_state_dict(checkpoint['state_dictS1'])
    modelS2.load_state_dict(checkpoint['state_dictS2'])
        
    print("=> loaded checkpoint '{}' (epoch {})".format(arguments.checkpoint_pth, checkpoint['epoch']))

    
    dataset = arguments.dataset
    trainedLabelsDir = os.path.join(dataset, 'trainedLabels.pt')
        
    fileTrainedS1Names = os.path.join(dataset, 'trainedS1Names.npy')
    fileTrainedS2Names = os.path.join(dataset, 'trainedS2Names.npy')
    
    fileGeneratedS1Codes = os.path.join(dataset, 'generatedS1Codes.pt')
    fileGeneratedS2Codes = os.path.join(dataset, 'generatedS2Codes.pt')
    
        
    trainedLabels = torch.load(trainedLabelsDir)
    trainedS1FileNames = np.load(fileTrainedS1Names)
    trainedS2FileNames = np.load(fileTrainedS2Names)
    trainedAndGeneratedS1Codes = torch.load(fileGeneratedS1Codes)
    trainedAndGeneratedS2Codes = torch.load(fileGeneratedS2Codes)




    modelS1.eval()
    modelS2.eval()

    label_test = []
    test_S1codes = []
    test_S2codes = []
    name_testS1 = []
    name_testS2 = []
    
    mapS1toS1 = 0 
    mapS1toS2 = 0
    mapS2toS1 = 0
    mapS2toS2 = 0
    
    totalSize = 0 
    


    with torch.no_grad():
        for batch_idx, (dataS1,dataS2) in enumerate(tqdm(test_data_loader, desc="test")):
            totalSize += dataS2["bands10"].size(0)
            
            if gpuDisabled :
                bands = torch.cat((dataS2["bands10"], dataS2["bands20"],dataS2["bands60"]), dim=1).to(torch.device("cpu"))
                polars = torch.cat((dataS1["polarVH"], dataS1["polarVV"]), dim=1).to(torch.device("cpu"))
                labels = dataS1["label"].to(torch.device("cpu"))  

            else:
                bands = torch.cat((dataS2["bands10"], dataS2["bands20"],dataS2["bands60"]), dim=1).to(torch.device("cuda"))
                polars = torch.cat((dataS1["polarVH"], dataS1["polarVV"]), dim=1).to(torch.device("cuda"))
                labels = dataS1["label"].to(torch.device("cuda"))           


            logitsS1 = modelS1(polars)
            logitsS2 = modelS2(bands)
            
            binaryS1 = torch.sigmoid(logitsS1)
            binaryS2 = torch.sigmoid(logitsS2)
        
            t = torch.Tensor([0.5])
            binaryS1 = (binaryS1 > t).float()
            binaryS2 = (binaryS2 > t).float()
            
            
                        
            test_S1codes += list(binaryS1)
            test_S2codes += list(binaryS2)
            label_test += list(labels)
            name_testS1 += list(dataS1['patchName'])
            name_testS2 += list(dataS2['patchName'])
                
            
            #S1 to S1
            neighboursIndices = get_k_hamming_neighbours(trainedAndGeneratedS1Codes, binaryS1)  
            mapPerBatch = get_mAP(neighboursIndices,arguments.k,trainedLabels,list(labels))
            mapS1toS1 += mapPerBatch
            
            #S1 to S2
            neighboursIndices = get_k_hamming_neighbours(trainedAndGeneratedS2Codes,binaryS1)  
            mapPerBatch = get_mAP(neighboursIndices,arguments.k,trainedLabels,list(labels))
            mapS1toS2 += mapPerBatch
            
            #S2 to S1
            neighboursIndices = get_k_hamming_neighbours(trainedAndGeneratedS1Codes,binaryS2)  
            mapPerBatch = get_mAP(neighboursIndices,arguments.k,trainedLabels,list(labels))
            mapS2toS1 += mapPerBatch
                  
            #S2 to S2
            neighboursIndices = get_k_hamming_neighbours(trainedAndGeneratedS2Codes, binaryS2)  
            mapPerBatch = get_mAP(neighboursIndices,arguments.k,trainedLabels,list(labels))
            mapS2toS2 += mapPerBatch
            

    print('Total Test Size: ',totalSize)
    mapS1toS1 = mapS1toS1 / totalSize * 100
    mapS1toS2 = mapS1toS2 / totalSize * 100
    mapS2toS1 = mapS2toS1 / totalSize * 100
    mapS2toS2 = mapS2toS2 / totalSize * 100

    print('MaP for S1 to S1: ', mapS1toS1)
    print('MaP for S1 to S2: ', mapS1toS2)
    print('MaP for S2 to S1: ', mapS2toS1)
    print('MaP for S2 to S2: ', mapS2toS2)



    averageMap = (mapS1toS1 + mapS1toS2 + mapS2toS1 + mapS2toS2 ) / 4
                   
    print('mAP@',arguments.k,':{0}'.format(averageMap))
    
        
    #An Example
    print('Testing S1 File Name: ', name_testS1[0])
    print('Testing Image Label Encoding: ', label_test[0])
    
    
    #S1 to S1
    neighboursIndices = get_k_hamming_neighbours(trainedAndGeneratedS1Codes, test_S1codes[0].reshape(1,-1))  
    firstImageIndex = neighboursIndices[0][0]
    secondIamgeIndex = neighboursIndices[0][1]

    print('#S1 to S1#')
    print('First Retrived S1 File Index: ', firstImageIndex)    
    print('First Retrived File Name: ',trainedS1FileNames[firstImageIndex])
    print('First Retrived Image Labels: ', trainedLabels[firstImageIndex])
    
    print('Second Retrived File Index: ',secondIamgeIndex)
    print('Second Retrived File Name: ',trainedS1FileNames[secondIamgeIndex])
    print('Second Retrived Image Labels: ', trainedLabels[secondIamgeIndex])
    
    #S1 to S2
    neighboursIndices = get_k_hamming_neighbours(trainedAndGeneratedS2Codes, test_S1codes[0].reshape(1,-1))  
    #neighboursIndices = get_k_hamming_neighbours(arguments.k,trainedAndGeneratedS2Codes,trainedLabels,trainedS2FileNames, list(test_S1codes[0].reshape(1,-1)), list(label_test[0]),list(name_testS1[0]),arguments.bits )  
    firstImageIndex = neighboursIndices[0][0]
    secondIamgeIndex = neighboursIndices[0][1]

    print('#S1 to S2#')
    print('First Retrived S1 File Index: ', firstImageIndex)    
    print('First Retrived File Name: ',trainedS2FileNames[firstImageIndex])
    print('First Retrived Image Labels: ', trainedLabels[firstImageIndex])
    
    print('Second Retrived File Index: ',secondIamgeIndex)
    print('Second Retrived File Name: ',trainedS2FileNames[secondIamgeIndex])
    print('Second Retrived Image Labels: ', trainedLabels[secondIamgeIndex])
    
    
    print('### Testing S2 File Name:  ### ', name_testS2[0])
    print('Testing Image Label Encoding: ', label_test[0])
    
    
    #S2 to S1
    neighboursIndices = get_k_hamming_neighbours(trainedAndGeneratedS1Codes, test_S2codes[0].reshape(1,-1))  
    #neighboursIndices = get_k_hamming_neighbours(arguments.k,trainedAndGeneratedS1Codes,trainedLabels,trainedS1FileNames, list(test_S2codes[0].reshape(1,-1)), list(label_test[0]),list(name_testS2[0]),arguments.bits )  
    firstImageIndex = neighboursIndices[0][0]
    secondIamgeIndex = neighboursIndices[0][1]

    print('#S2 to S1#')
    print('First Retrived S1 File Index: ', firstImageIndex)    
    print('First Retrived File Name: ',trainedS1FileNames[firstImageIndex])
    print('First Retrived Image Labels: ', trainedLabels[firstImageIndex])
    
    print('Second Retrived File Index: ',secondIamgeIndex)
    print('Second Retrived File Name: ',trainedS1FileNames[secondIamgeIndex])
    print('Second Retrived Image Labels: ', trainedLabels[secondIamgeIndex])
    
    #S2 to S2
    neighboursIndices = get_k_hamming_neighbours(trainedAndGeneratedS2Codes, test_S2codes[0].reshape(1,-1))  
    #neighboursIndices = get_k_hamming_neighbours(arguments.k,trainedAndGeneratedS2Codes,trainedLabels,trainedS2FileNames, list(test_S2codes[0].reshape(1,-1)), list(label_test[0]),list(name_testS2[0]),arguments.bits )  
    firstImageIndex = neighboursIndices[0][0]
    secondIamgeIndex = neighboursIndices[0][1]

    print('#S2 to S2#')
    print('First Retrived S1 File Index: ', firstImageIndex)    
    print('First Retrived File Name: ',trainedS2FileNames[firstImageIndex])
    print('First Retrived Image Labels: ', trainedLabels[firstImageIndex])
    
    print('Second Retrived File Index: ',secondIamgeIndex)
    print('Second Retrived File Name: ',trainedS2FileNames[secondIamgeIndex])
    print('Second Retrived Image Labels: ', trainedLabels[secondIamgeIndex])
    
    
    

if __name__ == "__main__":
    main()
