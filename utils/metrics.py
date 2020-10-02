""" 
metrics utilized for evaluating multi-label classification system
"""
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

from sklearn.metrics import f1_score, precision_score, recall_score, fbeta_score, \
    classification_report, hamming_loss, accuracy_score, coverage_error, label_ranking_loss,\
    label_ranking_average_precision_score, classification_report


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


class Precision_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):

        sample_prec = precision_score(true_labels, predict_labels, average='samples')
        micro_prec = precision_score(true_labels, predict_labels, average='micro')
        macro_prec = precision_score(true_labels, predict_labels, average='macro')

        return macro_prec, micro_prec, sample_prec    


class Recall_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):

        sample_rec = recall_score(true_labels, predict_labels, average='samples')
        micro_rec = recall_score(true_labels, predict_labels, average='micro')
        macro_rec = recall_score(true_labels, predict_labels, average='macro')

        return macro_rec, micro_rec, sample_rec


class F1_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):

        macro_f1 = f1_score(true_labels, predict_labels, average="macro")
        micro_f1 = f1_score(true_labels, predict_labels, average="micro")
        sample_f1 = f1_score(true_labels, predict_labels, average="samples")

        return macro_f1, micro_f1, sample_f1


class F2_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):

        macro_f2 = fbeta_score(true_labels, predict_labels, beta=2, average="macro")
        micro_f2 = fbeta_score(true_labels, predict_labels, beta=2, average="micro")
        sample_f2 = fbeta_score(true_labels, predict_labels, beta=2, average="samples")

        return macro_f2, micro_f2, sample_f2

class Hamming_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):

        return hamming_loss(true_labels, predict_labels)

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




def get_k_hamming_neighbours(enc_train,enc_query_imgs):
    
    
    hammingDistances = torch.cdist(enc_query_imgs,enc_train, p = 0)
    #hammingDistances = torch.cdist(queryCodeTensors,trainedCodeTensors, p = 2)
    sortedDistances, indices = torch.sort(hammingDistances)
    
    return indices
    '''
    if torch.cuda.is_available():
        return indices
    else
    return indices.detach().cpu().numpy().astype('int')
    '''
    
def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))    
    
    


class Subset_accuracy(nn.Module): 

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):

        return accuracy_score(true_labels, predict_labels)

class Accuracy_score(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, predict_labels, true_labels):

        # sample accuracy
        TP = (np.logical_and((predict_labels == 1), (true_labels == 1))).astype(int)
        union = (np.logical_or((predict_labels == 1), (true_labels == 1))).astype(int)
        TP_sample = TP.sum(axis=1)
        union_sample = union.sum(axis=1)

        sample_Acc = TP_sample/union_sample

        assert np.isfinite(sample_Acc).all(), 'Nan found in sample accuracy'

        FP = (np.logical_and((predict_labels == 1), (true_labels == 0))).astype(int)
        TN = (np.logical_and((predict_labels == 0), (true_labels == 0))).astype(int)
        FN = (np.logical_and((predict_labels == 0), (true_labels == 1))).astype(int)

        TP_cls = TP.sum(axis=0)
        FP_cls = FP.sum(axis=0)
        TN_cls = TN.sum(axis=0)
        FN_cls = FN.sum(axis=0)

        assert (TP_cls+FP_cls+TN_cls+FN_cls == predict_labels.shape[0]).all(), 'wrong'

        macro_Acc = np.mean((TP_cls + TN_cls) / (TP_cls + FP_cls + TN_cls + FN_cls))

        micro_Acc = (TP_cls.mean() + TN_cls.mean()) / (TP_cls.mean() + FP_cls.mean() + TN_cls.mean() + FN_cls.mean())

        return macro_Acc, micro_Acc, sample_Acc.mean()


class One_error(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict_probs, true_labels):
        row_inds = np.arange(predict_probs.shape[0])
        col_inds = np.argmax(predict_probs, axis=1)
        return np.mean((true_labels[tuple(row_inds), tuple(col_inds)] == 0).astype(int))

class Coverage_error(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_probs, true_labels):
        return coverage_error(true_labels, predict_probs)

class Ranking_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_probs, true_labels):
        return label_ranking_loss(true_labels, predict_probs)

class LabelAvgPrec_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_probs, true_labels):
        return label_ranking_average_precision_score(true_labels, predict_probs)

class calssification_report(nn.Module):

    def __init__(self, target_names):
        super().__init__()
        self.target_names = target_names
    def forward(self, predict_labels, true_labels):

        report = classification_report(true_labels, predict_labels, target_names=self.target_names, output_dict=True)

        return report

if __name__ == "__main__":
    acc = Accuracy_score()

    aa = (np.random.randn(100,20)>=0).astype(int)

    bb = (np.random.randn(100,20)>=0).astype(int)

    samp_acc, macro_acc, micro_acc = acc(aa, bb)

    print(samp_acc)
    print(macro_acc)
    print(micro_acc)











