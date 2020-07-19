# -*- coding: utf-8 -*-
#
# This script can be used to train any deep learning model on the BigEarthNet.
#
# To run the code, you need to provide a json file for configurations of the training.
#
# Author: Jian Kang, https://www.rsim.tu-berlin.de/menue/team/dring_jian_kang/
# Email: jian.kang@tu-berlin.de

import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import shutil

import argparse
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn as nn

import sys
sys.path.append('../')

from utils.ResNet import ResNet50_S1, ResNet50_S2

from utils.dataGenBigEarth import dataGenBigEarthLMDB, ToTensor, Normalize, ConcatDataset

from utils.metrics import MetricTracker, get_k_hamming_neighbours, get_mAP, Recall_score, F1_score, F2_score, Hamming_loss, Subset_accuracy, \
    Accuracy_score, One_error, Coverage_error, Ranking_loss, LabelAvgPrec_score


parser = argparse.ArgumentParser(description='PyTorch multi-label classification')
parser.add_argument('--S1LMDBPth', metavar='DATA_DIR',
                        help='path to the saved sentinel 1 LMDB dataset')
parser.add_argument('--S2LMDBPth', metavar='DATA_DIR',
                        help='path to the saved sentinel 2 LMDB dataset')
parser.add_argument('-b', '--batch-size', default=200, type=int,
                        metavar='N', help='mini-batch size (default: 200)')
parser.add_argument('--epochs', type=int, default=500, help='epoch number')
parser.add_argument('--k', type=int, default=20, help='number of retrived images per query')

parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--resume', '-r', help='path to the pretrained weights file', default=None, type=str)
parser.add_argument('--num_workers', default=8, type=int, metavar='N',
                        help='num_workers for data loading in pytorch')

parser.add_argument('--bits', type=int, default=16, help='number of bits to use in hashing')

parser.add_argument('--BigEarthNet19', dest='BigEarthNet19', action='store_true',
                    help='use the BigEarthNet19 class nomenclature')
parser.add_argument('--big1000', dest='big1000', action='store_true', help='for small dataset')
parser.add_argument('--serbia', dest='serbia', action='store_true',
                    help='use the serbia patches')

parser.add_argument('--train_csvS1', metavar='CSV_PTH',
                        help='path to the csv file of train patches')
parser.add_argument('--val_csvS1', metavar='CSV_PTH',
                        help='path to the csv file of val patches')
parser.add_argument('--test_csvS1', metavar='CSV_PTH',
                        help='path to the csv file of test patches')





args = parser.parse_args()



checkpoint_dir = os.path.join('./', 'Resnet50Pair', 'checkpoints')
logs_dir = os.path.join('./', 'Resnet50Pair', 'logs')
dataset_dir = os.path.join('./','Resnet50Pair','dataset')


if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.isdir(logs_dir):
    os.makedirs(logs_dir)
if not os.path.isdir(dataset_dir):
    os.makedirs(dataset_dir)
    
def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in vars(args).items():
            f.write('%s: %s\n' % (key, str(value)))

def save_checkpoint(state, is_best, name):

    filename = os.path.join(checkpoint_dir, name + '_checkpoint.pth.tar')

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, name + '_model_best.pth.tar'))



def main():
    global args

    sv_name = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    print('saving file name is ', sv_name)

    write_arguments_to_file(args, os.path.join(logs_dir, sv_name+'_arguments.txt'))

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True

    if args.serbia:
        if args.big1000:
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
            #Sentinel 1 in Serbia Statistics
            polars_mean = {
                    'polarVH_mean': [ -15.827944 ],
                    'polarVV_mean': [ -9.317011]
                }

            polars_std = {
                    'polarVH_std': [ 0.782826 ],
                    'polarVV_std': [ 1.8147297]
            }
            
            
            
            
            
            
    else:
        if args.big1000 :
            
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
                
                


    modelS1 = ResNet50_S1(args.bits)
    modelS2 = ResNet50_S2(args.bits)
    
    
    train_dataGenS1 =  dataGenBigEarthLMDB(
                    bigEarthPthLMDB=args.S1LMDBPth,
                    isSentinel2 = False,
                    state='train',
                    imgTransform=transforms.Compose([
                        ToTensor(isSentinel2 = False),
                        Normalize(polars_mean, polars_std, False)
                    ]),
                    upsampling=False,
                    train_csv=args.train_csvS1,
                    val_csv=args.val_csvS1,
                    test_csv=args.test_csvS1
    )
        
    train_dataGenS2 = dataGenBigEarthLMDB(
                    bigEarthPthLMDB=args.S2LMDBPth,
                    isSentinel2 = True,
                    state='train',
                    imgTransform=transforms.Compose([
                        ToTensor(isSentinel2=True),
                        Normalize(bands_mean, bands_std,True)
                    ]),
                    upsampling=True,
                    train_csv=args.train_csvS1,
                    val_csv=args.val_csvS1,
                    test_csv=args.test_csvS1
    )

    val_dataGenS1 = dataGenBigEarthLMDB(
                    bigEarthPthLMDB=args.S1LMDBPth,
                    isSentinel2 = False,
                    state='val',
                    imgTransform=transforms.Compose([
                        ToTensor(False),
                        Normalize(polars_mean, polars_std,False)
                    ]),
                    upsampling=False,
                    train_csv=args.train_csvS1,
                    val_csv=args.val_csvS1,
                    test_csv=args.test_csvS1
    )
    
    val_dataGenS2 = dataGenBigEarthLMDB(
                    bigEarthPthLMDB=args.S2LMDBPth,
                    isSentinel2 = True,
                    state='val',
                    imgTransform=transforms.Compose([
                        ToTensor(True),
                        Normalize(bands_mean, bands_std,True)
                    ]),
                    upsampling=True,
                    train_csv=args.train_csvS1,
                    val_csv=args.val_csvS1,
                    test_csv=args.test_csvS1
    )
    
    train_data_loader = DataLoader(
            ConcatDataset(train_dataGenS1,train_dataGenS2),
            batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)

    val_data_loader = DataLoader(
            ConcatDataset(val_dataGenS1,val_dataGenS2),
            batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)    
    
    
    #train_data_loader = DataLoader(train_dataGen, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    #val_data_loader = DataLoader(val_dataGen, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    if torch.cuda.is_available():
        modelS1.cuda()
        modelS2.cuda()
        gpuDisabled = False
    else:
        modelS1.cpu()
        modelS2.cpu()
        gpuDisabled = True
        
    print('GPU Disabled: ',gpuDisabled)
    

    lossFunc = nn.MSELoss()
    

    optimizerS1 = optim.Adam(modelS1.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizerS2 = optim.Adam(modelS2.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = 0

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            checkpoint_nm = os.path.basename(args.resume)
            sv_name = checkpoint_nm.split('_')[0] + '_' + checkpoint_nm.split('_')[1]
            print('saving file name is ', sv_name)

            if checkpoint['epoch'] > start_epoch:
                start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_map']
            modelS1.load_state_dict(checkpoint['state_dictS1'])
            modelS2.load_state_dict(checkpoint['state_dictS2'])
            optimizerS1.load_state_dict(checkpoint['optimizerS1'])
            optimizerS2.load_state_dict(checkpoint['optimizerS2'])

            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'training'))
    val_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'val'))


    toWrite = False
 
    for epoch in range(start_epoch, args.epochs):

        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)


        generatedCodesS1,generatedCodesS2, trainedLabels, name_trainS1, name_trainS2 = train(train_data_loader, modelS1,modelS2, optimizerS1,optimizerS2, epoch, use_cuda, train_writer,gpuDisabled,lossFunc)
        
        # To binarize s1 and s2 codes
        generatedCodesS1 = torch.stack(generatedCodesS1).reshape(len(generatedCodesS1),args.bits)
        generatedCodesS2 = torch.stack(generatedCodesS2).reshape(len(generatedCodesS2),args.bits)
        
        generatedCodesS1 = torch.sigmoid(generatedCodesS1)
        generatedCodesS2 = torch.sigmoid(generatedCodesS2)
        
        
        if gpuDisabled:
            t = torch.Tensor([0.5])
        else:
            t = torch.cuda.FloatTensor(1).fill_(0.5)

            
        generatedCodesS1 = (generatedCodesS1 >= t).float()
        generatedCodesS2 = (generatedCodesS2 >= t).float()
                
        mAP,val_S1codes,val_S2codes,label_val,name_valS1,name_valS2 = val(val_data_loader, modelS1,modelS2, optimizerS1,optimizerS2, val_writer,generatedCodesS1,generatedCodesS2, trainedLabels, name_trainS1,name_trainS2,gpuDisabled)
        val_S1codes = torch.stack(val_S1codes).reshape(len(val_S1codes),args.bits)
        val_S2codes = torch.stack(val_S2codes).reshape(len(val_S2codes),args.bits)





        is_best_acc = mAP > best_acc
        best_acc = max(best_acc, mAP)
        
        print('is_best_acc: ',is_best_acc)

        if is_best_acc:
            
            generatedS1CodesToWriteFile = []
            generatedS2CodesToWriteFile = []
            trainedLabelsToWriteFile = []
            trainedS1FileNamesToWriteFile = []    
            trainedS2FileNamesToWriteFile = []  
            
            
            generatedS1CodesToWriteFile = torch.cat((generatedCodesS1 , val_S1codes))
            generatedS2CodesToWriteFile = torch.cat((generatedCodesS2 , val_S2codes))
            
            trainedLabelsToWriteFile = trainedLabels + label_val
            
            trainedS1FileNamesToWriteFile = name_trainS1 + name_valS1
            trainedS2FileNamesToWriteFile = name_trainS2 + name_valS2

            epochToWrite = epoch
            stateS1DictToWrite = modelS1.state_dict()
            stateS2DictToWrite = modelS2.state_dict()

            optimizerS1ToWrite = optimizerS1.state_dict()
            optimizerS2ToWrite = optimizerS2.state_dict()

            bestMapToWrite = best_acc
            toWrite = True


    
    
    if toWrite :
        
        
        dataset_folder = os.path.join(dataset_dir,sv_name)
        if not os.path.isdir(dataset_folder):
            os.makedirs(dataset_folder)
        
        fileGeneratedS1Codes = os.path.join(dataset_folder, 'generatedS1Codes.pt')
        fileGeneratedS2Codes = os.path.join(dataset_folder, 'generatedS2Codes.pt')

        fileTrainedS1Names = os.path.join(dataset_folder, 'trainedS1Names.npy')
        fileTrainedS2Names = os.path.join(dataset_folder, 'trainedS2Names.npy')

        fileTrainedLabels = os.path.join(dataset_folder,'trainedLabels.pt')


        
        torch.save(generatedS1CodesToWriteFile,fileGeneratedS1Codes)
        torch.save(generatedS2CodesToWriteFile,fileGeneratedS2Codes)
        
        np.save(fileTrainedS1Names, trainedS1FileNamesToWriteFile )
        np.save(fileTrainedS2Names, trainedS2FileNamesToWriteFile )

        torch.save(trainedLabelsToWriteFile, fileTrainedLabels)
        
        save_checkpoint({
            'epoch': epochToWrite,
            'state_dictS1': stateS1DictToWrite,
            'state_dictS2': stateS2DictToWrite,
            'optimizerS1': optimizerS1ToWrite,
            'optimizerS2': optimizerS2ToWrite,
            'best_map': bestMapToWrite,
        }, True,sv_name)
        
        
    


def train(trainloader, modelS1,modelS2, optimizerS1,optimizerS2, epoch, use_cuda, train_writer,gpuDisabled,lossFunc):

    label_train = []
    name_trainS1 = []
    name_trainS2 = []
    generated_S1codes = []
    generated_S2codes = []    
    
    lossTracker = MetricTracker()
    
    modelS1.train()
    modelS2.train()
    
    for idx, (dataS1,dataS2) in enumerate(tqdm(trainloader, desc="training")):
        
        numSample = dataS2["bands10"].size(0)
        halfNumSample = numSample // 2
        
        #print('halfNumSample:',halfNumSample)

        if gpuDisabled :
            bands1 = torch.cat((dataS2["bands10"][:halfNumSample], dataS2["bands20"][:halfNumSample],dataS2["bands60"][:halfNumSample]), dim=1).to(torch.device("cpu"))
            polars1 = torch.cat((dataS1["polarVH"][:halfNumSample], dataS1["polarVV"][:halfNumSample]), dim=1).to(torch.device("cpu"))
            labels1 = dataS2["label"][:halfNumSample].to(torch.device("cpu")) 
            
            bands2 = torch.cat((dataS2["bands10"][halfNumSample:], dataS2["bands20"][halfNumSample:],dataS2["bands60"][halfNumSample:]), dim=1).to(torch.device("cpu"))
            polars2 = torch.cat((dataS1["polarVH"][halfNumSample:], dataS1["polarVV"][halfNumSample:]), dim=1).to(torch.device("cpu"))
            labels2 = dataS2["label"][halfNumSample:].to(torch.device("cpu")) 
            
            onesTensor = torch.ones(halfNumSample)
            
        else:            
            bands1 = torch.cat((dataS2["bands10"][:halfNumSample], dataS2["bands20"][:halfNumSample],dataS2["bands60"][:halfNumSample]), dim=1).to(torch.device("cuda"))
            polars1 = torch.cat((dataS1["polarVH"][:halfNumSample], dataS1["polarVV"][:halfNumSample]), dim=1).to(torch.device("cuda"))
            labels1 = dataS2["label"][:halfNumSample].to(torch.device("cuda")) 
            
            bands2 = torch.cat((dataS2["bands10"][halfNumSample:], dataS2["bands20"][halfNumSample:],dataS2["bands60"][halfNumSample:]), dim=1).to(torch.device("cuda"))
            polars2 = torch.cat((dataS1["polarVH"][halfNumSample:], dataS1["polarVV"][halfNumSample:]), dim=1).to(torch.device("cuda"))
            labels2 = dataS2["label"][halfNumSample:].to(torch.device("cuda")) 
            
            onesTensor = torch.cuda.FloatTensor(halfNumSample).fill_(0)
        
                
        
        optimizerS1.zero_grad()
        optimizerS2.zero_grad()
        
        logitsS1_1 = modelS1(polars1)
        logitsS1_2 = modelS1(polars2)
        
        logitsS2_1 = modelS2(bands1)
        logitsS2_2 = modelS2(bands2)


        cos = torch.nn.CosineSimilarity(dim=1)
        cosBetweenLabels = cos(labels1,labels2)
        
        cosBetweenS1 = cos(logitsS1_1,logitsS1_2)
        cosBetweenS2 = cos(logitsS2_1,logitsS2_2)
        
        cosInterSameLabel1 = cos(logitsS1_1,logitsS2_1  )
        cosInterSameLabel2 = cos(logitsS1_2, logitsS2_2)
        
        cosInterDifLabel1 = cos(logitsS1_1,logitsS2_2)
        cosInterDifLabel2 = cos(logitsS1_2,logitsS2_1)
        
    
        S1IntraLoss = lossFunc(cosBetweenS1,cosBetweenLabels)
        S2IntraLoss = lossFunc(cosBetweenS2,cosBetweenLabels)
    

        InterLoss_SameLabel1 = lossFunc(cosInterSameLabel1,onesTensor)
        InterLoss_SameLabel2 = lossFunc(cosInterSameLabel2,onesTensor)
        
        InterLoss_DifLabel1 = lossFunc(cosInterDifLabel1,cosBetweenLabels)
        InterLoss_DifLabel2 = lossFunc(cosInterDifLabel2,cosBetweenLabels)
        
        loss = 0.33 * S1IntraLoss + 0.33 * S2IntraLoss + 0.0825 * InterLoss_SameLabel1 + 0.0825 *  InterLoss_SameLabel2 + 0.0825 * InterLoss_DifLabel1 * 0.0825 * InterLoss_DifLabel2
        

        

        loss.backward()
        optimizerS1.step()
        optimizerS2.step()
        

        lossTracker.update(loss.item(), numSample)

        generated_S1codes += list(logitsS1_1) + list(logitsS1_2)
        generated_S2codes += list(logitsS2_1) + list(logitsS2_2)
        
        label_train += list(dataS1["label"])
        
        name_trainS1 += list(dataS1['patchName'])
        name_trainS2 += list(dataS2['patchName'])
        

    train_writer.add_scalar("loss", lossTracker.avg, epoch)

    print('Train loss: {:.6f}'.format(lossTracker.avg))
    
    
    return (generated_S1codes,generated_S2codes,label_train,name_trainS1,name_trainS2)


def val(valloader, modelS1,modelS2, optimizerS1,optimizerS2, val_writer,generatedCodesS1,generatedCodesS2, trainedLabels,name_trainS1,name_trainS2,gpuDisabled):


    modelS1.eval()
    modelS2.eval()

    label_val = []
    predicted_S1codes = []
    predicted_S2codes = []
    name_valS1 = []
    name_valS2 = []
    
    mapS1toS1 = 0 
    mapS1toS2 = 0
    mapS2toS1 = 0
    mapS2toS2 = 0
    
    totalSize = 0 

    with torch.no_grad():
        for batch_idx, (dataS1,dataS2) in enumerate(tqdm(valloader, desc="validation")):

            totalSize += dataS2["bands10"].size(0)
            
            if gpuDisabled:
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
        
        
            if gpuDisabled:
                t = torch.Tensor([0.5])
            else:
                t = torch.cuda.FloatTensor(1).fill_(0.5)
        
            binaryS1 = (binaryS1 > t).float()
            binaryS2 = (binaryS2 > t).float()



            predicted_S1codes += list(binaryS1)
            predicted_S2codes += list(binaryS2)
            
            
            label_val += list(labels)
            name_valS1 += list(dataS1['patchName'])
            name_valS2 += list(dataS2['patchName'])
            
            #S1 to S1
            neighboursIndices = get_k_hamming_neighbours(generatedCodesS1, binaryS1)  
            mapPerBatch = get_mAP(neighboursIndices,args.k,trainedLabels,list(labels))
            mapS1toS1 += mapPerBatch
            
            #S1 to S2
            neighboursIndices = get_k_hamming_neighbours(generatedCodesS2,binaryS1)  
            mapPerBatch = get_mAP(neighboursIndices,args.k,trainedLabels,list(labels))
            mapS1toS2 += mapPerBatch
            
            #S2 to S1
            neighboursIndices = get_k_hamming_neighbours(generatedCodesS1,binaryS2)  
            mapPerBatch = get_mAP(neighboursIndices,args.k,trainedLabels,list(labels))
            mapS2toS1 += mapPerBatch
                  
            #S2 to S2
            neighboursIndices = get_k_hamming_neighbours(generatedCodesS2, binaryS2)  
            mapPerBatch = get_mAP(neighboursIndices,args.k,trainedLabels,list(labels))
            mapS2toS2 += mapPerBatch
    
    #print('MapS1toS1: ',mapS1toS1)
    #print('len: ',totalSize)
    
    mapS1toS1 = mapS1toS1 / totalSize * 100
    mapS1toS2 = mapS1toS2 / totalSize * 100
    mapS2toS1 = mapS2toS1 / totalSize * 100
    mapS2toS2 = mapS2toS2 / totalSize * 100

    print('MaP for S1 to S1: ', mapS1toS1)
    print('MaP for S1 to S2: ', mapS1toS2)
    print('MaP for S2 to S1: ', mapS2toS1)
    print('MaP for S2 to S2: ', mapS2toS2)



    averageMap = (mapS1toS1 + mapS1toS2 + mapS2toS1 + mapS2toS2 ) / 4
                   
    print('mAP@',args.k,':{0}'.format(averageMap))
    return (averageMap,predicted_S1codes,predicted_S2codes,label_val,name_valS1,name_valS2)


if __name__ == "__main__":
    main()
    
    
 
    
    
    
    