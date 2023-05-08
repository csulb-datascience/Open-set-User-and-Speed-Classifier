import numpy as np
import pandas as pd
import re
import os
import pickle
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch
from torch.nn import DataParallel
from torch.nn.parameter import Parameter
import math
import random
from matplotlib import pyplot as plt
from tqdm import trange, tqdm
import time
import os
from data import Dataset
from torch.utils import data
from models import *
import torchvision
from config import Config
from torch.optim.lr_scheduler import StepLR
from test import *
from model_class import *
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn import metrics


N_PEOPLE = 14
N_SPEED = 5
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class ArcMarginProduct_(nn.Module):
    
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct_, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        
        nn.init.xavier_uniform_(self.weight)
        
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input.to(device)), F.normalize(self.weight.to(device)))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size()).to(device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

def split_dataset(dataset, personID, speedID):

    ## select 10 random numbers for train, 4 for known test, 4 for unknown test
    from random import shuffle
    shuffleOrder = list(range(N_PEOPLE))
    shuffle(shuffleOrder)
    # print(shuffleOrder)
    x_train = []
    y1_train = []
    y2_train = []
    x_ktest = []
    y1_ktest = []
    y2_ktest = []
    x_utest = []
    y1_utest = []
    y2_utest = []
    for i in range(6):
        x_train.append(dataset[shuffleOrder[i]])
        y2_train.append(speedID[shuffleOrder[i]])
    for i in range(6,10):
        x_ktest.append(dataset[shuffleOrder[i]])
        y2_ktest.append(speedID[shuffleOrder[i]])
    for i in range(10,14):
        x_utest.append(dataset[shuffleOrder[i]])
        y2_utest.append(speedID[shuffleOrder[i]])
    y1_train = [[i] * len(x_train[i]) for i in range(len(x_train))]
    y1_ktest = [[i] * len(x_ktest[i]) for i in range(len(x_ktest))]
    y1_utest = [[i] * len(x_utest[i]) for i in range(len(x_utest))]
    return (x_train, y1_train, y2_train), (x_ktest, y1_ktest, y2_ktest), (x_utest, y1_utest, y2_utest)
    
def split_small_large_set(x_ktest, y1_ktest, y2_ktest, n_select = 50):
    ## Split known tests into small = 50 and large
    x_ktest_small = []
    y1_ktest_small = []
    y2_ktest_small = []
    x_ktest_large = []
    y1_ktest_large = []
    y2_ktest_large = []
    
    for x,y1,y2 in zip(x_ktest,y1_ktest,y2_ktest):
        
        num_list = random.sample(range(len(x)), n_select)
        x_ktest_small.append(list(map(x.__getitem__, num_list)))
        y1_ktest_small.append(list(map(y1.__getitem__, num_list)))
        y2_ktest_small.append(list(map(y2.__getitem__, num_list)))
        x_ktest_large.extend(x)
        y1_ktest_large.extend(y1)
        y2_ktest_large.extend(y2)
        
    
    x_ktest_large = torch.as_tensor(np.array(x_ktest_large), dtype=torch.float)
    x_ktest_small = torch.as_tensor(np.array(x_ktest_small), dtype=torch.float)
    
    y1_ktest_large = torch.as_tensor(np.array(y1_ktest_large), dtype = torch.long)
    y1_ktest_small = torch.as_tensor(np.array(y1_ktest_small), dtype = torch.long)
    
    y2_ktest_large = torch.as_tensor(np.array(y2_ktest_large), dtype = torch.long)
    y2_ktest_small = torch.as_tensor(np.array(y2_ktest_small), dtype = torch.long)
    
    return (x_ktest_small, y1_ktest_small, y2_ktest_small), (x_ktest_large, y1_ktest_large, y2_ktest_large)

def convert_to_TensorDataloader(x, y, batch_size = 64, num_workers = 4):
    return DataLoader(TensorDataset(x, y), 
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers,
                      pin_memory=True, 
                      drop_last=True)

def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


def train(model, trainloader, num_classes, opt):

    print('{} train iters per epoch:'.format(len(trainloader)))
    # print(opt.optimizer)
    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=opt.gamma)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    
    if opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct_(512, num_classes, s=opt.s, m=opt.m, easy_margin=opt.easy_margin)
    else:
        metric_fc = nn.Linear(512, num_classes)
        
    print(model)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    start = time.time()
    for i in range(opt.max_epoch):
        
        torch.cuda.empty_cache()
        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device)
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))
                start = time.time()
        scheduler.step()
        
    return model

def plot_tsne(x,y, title = 'Plotted data', num_class = 8):

    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(x, y)
    df = pd.DataFrame()
    plt.figure()
    df["y"] = y
    df["comp-1"] = tsne_results[:,0]
    df["comp-2"] = tsne_results[:,1]
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", num_class),
                    data=df).set(title=title) 

def cosin_metric(x1, x2):
    def unit_vector(x):
        return x / np.linalg.norm(x)
    return np.arccos(np.clip(np.dot(unit_vector(x1), unit_vector(x2)), -1.0, 1.0))

def cal_accuracy_cosine(feat_model_large, centroid, y_true, theta):
    y_pred = []
    y_max = []
    n = len(centroid)
    for i in feat_model_large:
        calc = []
        for j in centroid:
            calc.append(cosin_metric(i,j))
        y_max.append(np.min(calc))
        if np.min(calc) <= theta:
            y_pred.append(np.argmin(calc))
        else:
            y_pred.append(n)
    print('Cosine metric scores:')
    print(len(y_true), len(y_pred))
    print('Accuracy: ', metrics.accuracy_score(y_true, y_pred))
    print('r2 score: ', metrics.r2_score(y_true, y_pred))
    print('Confusion matrix: ', metrics.confusion_matrix(y_true, y_pred)) 
    print('Classification report: ', metrics.classification_report(y_true, y_pred))
    
    return {
        'type': 'Cosine',
        'y_true': y_true,
        'y_pred' : y_pred,
        'y_max' : y_max,
        'Accuracy': metrics.accuracy_score(y_true, y_pred),
        'r2 score': metrics.r2_score(y_true, y_pred),
        'Confusion_matrix': metrics.confusion_matrix(y_true, y_pred),
        'Classification_report': metrics.classification_report(y_true, y_pred),
        'theta': theta
    }
          
    
def euc_dist(point1, point2):
    return np.linalg.norm(point1 - point2)

def model_train_eval(model, opt, num_classes, file_name, x_train_large, y1_train_large, x_ktest_small, y1_ktest_small, x_ktest_large, y1_ktest_large):
    train_l_dataloader_p = convert_to_TensorDataloader(x_train_large, y1_train_large, opt.train_batch_size, opt.num_workers)
    test_l_dataloader_p = convert_to_TensorDataloader(x_ktest_large, y1_ktest_large, 700, opt.num_workers)
    
    p_model = train(model, train_l_dataloader_p, num_classes = 6, opt = opt)
    torch.cuda.empty_cache()
    p_model.eval()
    centroid = []
    feats = []
    labels = []
    with torch.no_grad():
        for x in x_ktest_small:
            feat_model_ks = p_model(x)
            centroid.append(torch.mean(feat_model_ks, dim=0))
    
    centroid = torch.Tensor(torch.stack(centroid))
    
    with torch.no_grad():
        for ii, data in enumerate(test_l_dataloader_p):
            data_input, label = data
            feat_model_kl = p_model(data_input)
            feats.append(feat_model_kl)
            labels.append(label)
    
    feats = torch.cat(feats, dim = 0)
    labels = torch.cat(labels, dim = 0)
    print(feats.shape, labels.shape)
    acc_cos_outp = cal_accuracy_cosine(feats.cpu().detach().numpy(), centroid.cpu().detach().numpy(), labels.cpu().detach().numpy(), opt.cos_theta)
    acc_euc_outp = cal_accuracy_euc(feats.cpu().detach().numpy(), centroid.cpu().detach().numpy(), labels.cpu().detach().numpy(), opt.euc_theta)
    
    
    plot_tsne(feats.cpu().detach().numpy(), labels.numpy(), "Large test known person data T-SNE projection", 5)
    
    print('Range of cosine and eucledian metric')   
    print(min(acc_cos_outp['y_max']), max(acc_cos_outp['y_max']), min(acc_euc_outp['y_max']), max(acc_euc_outp['y_max']))
    
    fileData = 'ModelN/'+file_name+'.pkl'
    fileModel = 'ModelN/'+file_name+'.pt'
    torch.save(model, fileModel)
    import pickle 
    with open(fileData , 'wb') as f:
        pickle.dump((acc_cos_outp, acc_euc_outp), f, protocol = 4)


def model_eval_user(model, x_ktest_large, y1_ktest_large):
    test_l_dataloader_p = convert_to_TensorDataloader(x_ktest_large, y1_ktest_large, 500, 4)
    torch.cuda.empty_cache()
    model.eval()
    centroid = []
    feats = []
    labels = []
    with torch.no_grad():
        for ii, data in enumerate(test_l_dataloader_p):
            data_input, label = data
            feat_model_kl = model(data_input)
            feats.append(feat_model_kl)
            labels.append(label)
    
    feats = torch.cat(feats, dim = 0)
    labels = torch.cat(labels, dim = 0)
    plot_tsne(feats.cpu().detach().numpy(), labels.numpy(), "Large test known person data T-SNE projection", 7)

def model_eval_speed(model, x_ktest_large, y):
    test_l_dataloader_p = convert_to_TensorDataloader(x_ktest_large, y, 500, 4)
    torch.cuda.empty_cache()
    model.eval()
    centroid = []
    feats = []
    labels = []
    with torch.no_grad():
        for ii, data in enumerate(test_l_dataloader_p):
            data_input, label = data
            feat_model_kl = model(data_input)
            feats.append(feat_model_kl)
            labels.append(label)
    
    feats = torch.cat(feats, dim = 0)
    labels = torch.cat(labels, dim = 0)
    plot_tsne(feats.cpu().detach().numpy(), labels.numpy(), "Large test known speed data T-SNE projection", 5)

def cal_accuracy_euc(feat_model_large, centroid, y_true, theta):
    y_pred = []
    y_max = []
    n = len(centroid)
    for i in feat_model_large:
        calc = []
        for j in centroid:
            calc.append(euc_dist(i,j))
        y_max.append(np.min(calc))
        if np.min(calc) <= theta:
            y_pred.append(np.argmin(calc))
        else:
            y_pred.append(n)
    print('Eucledian metric scores:')
    print('Accuracy: ', metrics.accuracy_score(y_true, y_pred))
    print('r2 score: ', metrics.r2_score(y_true, y_pred))
    print('Confusion matrix: ', metrics.confusion_matrix(y_true, y_pred)) 
    print('Classification report: ', metrics.classification_report(y_true, y_pred))
    
    return {
        'type': 'Eucledian',
        'y_true': y_true,
        'y_pred' : y_pred,
        'y_max' : y_max,
        'Accuracy': metrics.accuracy_score(y_true, y_pred),
        'r2 score': metrics.r2_score(y_true, y_pred),
        'Confusion_matrix': metrics.confusion_matrix(y_true, y_pred),
        'Classification_report': metrics.classification_report(y_true, y_pred),
        'theta': theta
    }        
    