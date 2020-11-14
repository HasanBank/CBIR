""" 
metrics utilized for evaluating multi-label CBIR system
"""
import torch
import numpy as np
import os
import rasterio




class MetricTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


 

def get_mAP(indices, nrof_neighbors,trainLabels,queryLabels):
    
    totalMap = 0
    
    for i in range(len(indices)):
        acc = np.empty((0,)).astype(np.float)
        correct = 0
        
        if type(queryLabels) == list:
            queryLabel = queryLabels[i]
        elif type(queryLabels) == torch.Tensor:
            queryLabel = queryLabels
        
        for j in range(nrof_neighbors):            
            similarity = torch.sum(torch.mul(queryLabel,trainLabels[indices[i][j]])).ge(1.0)
            if similarity:
                correct += 1
                precision = (correct / float(j+1))
                acc = np.append(acc, [precision, ], axis=0)
            
        if correct == 0:
             singularMap = 0.
        else:
            num = np.sum(acc)
            den = correct
            singularMap =  num/den
        totalMap += singularMap
        
    return totalMap

def get_mAP_weighted(indices, nrof_neighbors,trainLabels,queryLabels):
    
    totalMap = 0
    
    for i in range(len(indices)):
        acgAverage = np.empty((0,)).astype(np.float)
        numberOfSharedLabels = 0
        correct = 0
        
        if type(queryLabels) == list:
            queryLabel = queryLabels[i]
        elif type(queryLabels) == torch.Tensor:
            queryLabel = queryLabels
        
        for j in range(nrof_neighbors):
            retrievedLabel = trainLabels[indices[i][j]]
            
            similarity = torch.sum(torch.mul(queryLabel,retrievedLabel))
            if similarity.ge(1.0):
                correct += 1
                numberOfSharedLabels = numberOfSharedLabels + similarity
                acg = (numberOfSharedLabels / float(j+1))
                acgAverage = np.append(acgAverage, [acg], axis=0)
                
                        
        if correct == 0:
             singularMap = 0.
        else:
            num = np.sum(acgAverage)
            den = correct
            singularMap =  num/den
        totalMap += singularMap
        
    return totalMap






def get_average_precision_recall(indices, nrof_neighbors, trainLabels, queryLabels):
    
    precisionTotal = np.empty((0,)).astype(np.float)
    recallTotal = np.empty((0,)).astype(np.float)
    
    precisionWeightedTotal = np.empty((0,)).astype(np.float)
    recallWeightedTotal = np.empty(0,).astype(np.float)
    
    for j in range(len(indices)):
        
        precisionPerQuery = np.empty((0,)).astype(np.float)
        recallPerQuery = np.empty((0,)).astype(np.float)
    
        precisionWeightedPerQuery = np.empty((0,)).astype(np.float)
        recallWeightedPerQuery = np.empty(0,).astype(np.float)
        

        if type(queryLabels) == list:
            queryLabel = queryLabels[j]
        elif type(queryLabels) == torch.Tensor:
            queryLabel = queryLabels
    
    
        for i in range(nrof_neighbors):
        
            retrivedImageLabel = trainLabels[indices[j][i]]
    
            numberOfLabels_retrived = torch.sum(retrivedImageLabel)
            numberOfLabels_query = torch.sum(queryLabel)
            numberOfCommonLabels = torch.sum(torch.mul(queryLabel,retrivedImageLabel))
            
            precisionPerInstance = numberOfCommonLabels / numberOfLabels_retrived
            recallPerInstance = numberOfCommonLabels / numberOfLabels_query
            
            precisionPerQuery = np.append(precisionPerQuery, [precisionPerInstance, ], axis=0)
            recallPerQuery = np.append(recallPerQuery, [recallPerInstance, ], axis=0)
            
            weightedPrecision = precisionPerInstance * (nrof_neighbors - i)
            weightedRecall = recallPerInstance * (nrof_neighbors - i)
            
            precisionWeightedPerQuery = np.append(precisionWeightedPerQuery, [weightedPrecision, ], axis=0)
            recallWeightedPerQuery = np.append(recallWeightedPerQuery, [weightedRecall, ], axis=0)
            
        
        precisionPerQuery = np.sum(precisionPerQuery) / nrof_neighbors
        recallPerQuery = np.sum(recallPerQuery) / nrof_neighbors

        totalWeights = (nrof_neighbors * (nrof_neighbors + 1) ) / 2
        precisionWeightedPerQuery = np.sum(precisionWeightedPerQuery) / totalWeights
        recallWeightedPerQuery = np.sum(recallWeightedPerQuery) / totalWeights
        
        precisionTotal = np.append(precisionTotal, [precisionPerQuery, ], axis = 0)
        recallTotal = np.append(recallTotal, [recallPerQuery, ], axis = 0)
        precisionWeightedTotal = np.append(precisionWeightedTotal, [precisionWeightedPerQuery, ], axis = 0)
        recallWeightedTotal = np.append(recallWeightedTotal, [recallWeightedPerQuery, ], axis = 0)

  
    return np.sum(precisionTotal), np.sum(recallTotal), np.sum(precisionWeightedTotal), np.sum(recallWeightedTotal)

    

def f1Score(precision,recall):
    return 2 * (precision * recall ) / ( precision + recall)

def calculateAverageMetric(sumMetric,totalSize):
    return sumMetric / totalSize * 100

def lineWriteToFile(fileName,lines):
    with open(fileName, 'a') as file:
        for line in lines:
            file.write(line)
        


def printResultsAndGetF1Scores(precision,precision_weighted,recall,recall_weighted,fromDataset,targetDataset,totalSize,resultsFile_name):
    
    precision = calculateAverageMetric(precision,totalSize)
    recall = calculateAverageMetric(recall,totalSize)
    precision_weighted = calculateAverageMetric(precision_weighted,totalSize)
    recall_weighted = calculateAverageMetric(recall_weighted,totalSize)
    
    f1 = f1Score(precision,recall)
    f1_weighted = f1Score(precision_weighted,recall_weighted)
    
    f1_string = tensorToStr(f1)
    f1_weighted_string = tensorToStr(f1_weighted)
    
    
    unweigtedMAPLine = "mAP {}-{}: {}\n ".format(fromDataset, targetDataset,tensorToStr(precision))
    weightedMAPLine = "mAP {}-{} (weighted): {}\n ".format(fromDataset, targetDataset,tensorToStr(precision_weighted))
    unweightedMARLine = "mAR {}-{}: {}\n ".format(fromDataset, targetDataset,tensorToStr(recall))
    weightedMARLine = "mAR {}-{} (weighted): {}\n ".format(fromDataset, targetDataset,tensorToStr(recall_weighted))
    unweightedF1Line = "f1 {}-{}: {}\n ".format(fromDataset, targetDataset,f1_string)
    weightedF1Line = 'f1 {}-{} (weighted): {}\n '.format(fromDataset, targetDataset,f1_weighted_string )
    
    lineWriteToFile(resultsFile_name,unweigtedMAPLine)
    lineWriteToFile(resultsFile_name,weightedMAPLine)
    lineWriteToFile(resultsFile_name,unweightedMARLine)
    lineWriteToFile(resultsFile_name,weightedMARLine)
    lineWriteToFile(resultsFile_name,unweightedF1Line)
    lineWriteToFile(resultsFile_name,weightedF1Line)


    print(unweightedF1Line)
    print(weightedF1Line)

    return (f1,f1_weighted)

def tensorToStr(tensor):
    if torch.cuda.is_available():
        return tensor.cpu().numpy().astype('str')
    else:
        return tensor.numpy().astype('str')
    




def get_k_hamming_neighbours(enc_train,enc_query_imgs):    
    hammingDistances = torch.cdist(enc_query_imgs,enc_train, p = 0)
    sortedDistances, indices = torch.sort(hammingDistances)    
    return indices


    
def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))    
    

def createTrueColorTiff(S2FolderDir,S2FolderName,destinationFileFullName):
    
    band2 = os.path.join(S2FolderDir,S2FolderName,S2FolderName+'_B02.tif')
    band3 = os.path.join(S2FolderDir,S2FolderName,S2FolderName+'_B03.tif')
    band4 = os.path.join(S2FolderDir,S2FolderName,S2FolderName+'_B04.tif')
    
    file_list = [band4, band3, band2]
    
    # Read metadata of first file
    with rasterio.open(file_list[0]) as src0:
        meta = src0.meta
    
    # Update meta to reflect the number of layers
    meta.update(count = len(file_list))
    
    # Read each layer and write it to stack
    with rasterio.open(destinationFileFullName, 'w', **meta) as dst:
        for id, layer in enumerate(file_list, start=1):
            with rasterio.open(layer) as src1:
                dst.write_band(id, src1.read(1))


#https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/product-overview/polarimetry
def falseRepresentationS1(S1FolderDir,S1FolderName,destinationFileFullName):
    
    polarVH = os.path.join(S1FolderDir,S1FolderName,S1FolderName+'_VH.tif')
    polarVV = os.path.join(S1FolderDir,S1FolderName,S1FolderName+'_VV.tif')
    
    #file_list = [polarVV, polarVH, polarVV/polarVH ]
    
    # Read metadata of first file
    with rasterio.open(polarVV) as src0:
        meta = src0.meta
    
    # Update meta to reflect the number of layers
    meta.update(count = 3)
    
    # Read each layer and write it to stack
    with rasterio.open(destinationFileFullName, 'w', **meta) as dst:
        #for id, layer in enumerate(file_list, start=1):
         #   with rasterio.open(layer) as src1:
          #      dst.write_band(id, src1.read(1))

        vhValues = rasterio.open(polarVH)
        vvValues = rasterio.open(polarVV)
        
    
        dst.write_band(1,vvValues.read(1))
        dst.write_band(2,vhValues.read(1))
        dst.write_band(3,vvValues.read(1) / vhValues.read(1) )
















