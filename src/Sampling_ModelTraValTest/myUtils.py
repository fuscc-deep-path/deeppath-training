import numpy as np
from tqdm import tqdm
import time
import setproctitle
import argparse
import json
import pandas as pd
from scipy import stats
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import os
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import myConfig
from sklearn.preprocessing import StandardScaler
import joblib

def my_standard_scale(data, testingflag, scaler_path):

    if testingflag:
        scaler = joblib.load(scaler_path)

    else:
        scaler = StandardScaler()
        scaler.fit(data)

        print(scaler.mean_)
        print(scaler.var_)
        joblib.dump(scaler, os.path.join(scaler_path))

    return scaler.transform(data)


def save_temp_excel(namelist, scores, predictions, reals, save_dir, nCls, PATCHorPATIENT, TRAINorVALorTEST):
    if nCls==2:
        b = pd.DataFrame({"namelist_" + PATCHorPATIENT + TRAINorVALorTEST: namelist,
                          "scores_" + PATCHorPATIENT + TRAINorVALorTEST: scores,
                          "predictions_" + PATCHorPATIENT + TRAINorVALorTEST: predictions,
                          "reals_"+ PATCHorPATIENT + TRAINorVALorTEST: reals})
    elif nCls==4:
        b = pd.DataFrame({"namelist_" + PATCHorPATIENT + TRAINorVALorTEST: namelist,
                          "predictions_" + PATCHorPATIENT + TRAINorVALorTEST: predictions,
                          "reals_" + PATCHorPATIENT + TRAINorVALorTEST: reals})
    elif nCls==1:
        b = pd.DataFrame({"namelist_" + PATCHorPATIENT + TRAINorVALorTEST: namelist,
                          "scores_" + PATCHorPATIENT + TRAINorVALorTEST: scores,
                          "reals_"+ PATCHorPATIENT + TRAINorVALorTEST: reals})

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    b.to_excel(os.path.join(save_dir, PATCHorPATIENT+TRAINorVALorTEST+'.xlsx'))


def AdjustLR(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def Train(loader, network, optimizer, criterion, epoch, iterations, b): # arc = 'inception'
    start = time.time()

    losses = 0.0
    network.train()
    for i, (img, label, _) in enumerate(loader):
        img = img.cuda()
        label = label.cuda().long()

        output = network(img)
        loss = criterion(output, label)
        loss = (loss-b).abs() + b
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()

        print('Iteration {:3d} loss {:.6f}'.format(i + 1, loss.item()))
        setproctitle.setproctitle("Iteration:{}/{}".format(i + 1, iterations))

    avgloss = losses/(i + 1)
    print('Epoch{:3d}--Time(s){:.2f}--Avgloss{:.4f}-'.format(epoch, time.time() - start, avgloss))
    return network, avgloss


def RocPlot(real, score, figname=None, font={'size': 20}):
    fpr, tpr, thresholds = roc_curve(real, score, pos_label=1, drop_intermediate=False)  # calculate fpr and tpr
    youden_index = [tpr[i] - fpr[i] for i in range(len(fpr))]
    threshold_YI = min(thresholds[youden_index == (max(youden_index))])
    AUC = auc(fpr, tpr)  # for AUC value

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='red', lw=3, label=None)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    #plt.legend(loc="lower right", prop=font)
    if figname is not None:
        plt.savefig(figname, dpi=600, quality=95)
    plt.show()
    plt.close('all')
    return AUC, threshold_YI


# def NetPrediction(dataloader, model, arc=None): # inception_v3
#     reals = np.array([])
#     scores = np.array([])
#     # predictions = np.array([])
#     namelist = np.array([])
#
#     model.eval()
#     with torch.no_grad():
#         for i, (img, target, name) in tqdm(enumerate(dataloader)):
#             out = model(img.cuda())
#
#             prob = F.softmax(out, 1)
#             # pred = torch.argmax(prob, dim=1)
#             reals = np.append(reals, target)
#             # predictions  = np.concatenate((predictions, pred.cpu().numpy()), axis=0)
#             scores = np.append(scores, prob.cpu().numpy()[:, 1])
#             namelist = np.append(namelist, name)
#     return reals, scores, namelist


def NetPrediction(dataloader, model, Cls):
    reals = np.array([])
    scores = np.empty([0, Cls])
    predictions = np.array([])
    namelist = np.array([])

    model.eval()
    with torch.no_grad():
        for i, (img, target, name) in tqdm(enumerate(dataloader)):
            out = model(img.cuda())

            prob = F.softmax(out, 1, _stacklevel=5)
            pred = torch.argmax(prob, dim=1)

            reals = np.concatenate((reals, target), axis=0)
            predictions = np.concatenate((predictions, pred.cpu().numpy()), axis=0)
            scores = np.concatenate((scores, prob.cpu().numpy()), axis=0)
            scores = np.reshape(scores, (-1, Cls))
            namelist = np.concatenate((namelist, name), axis=0)

    return reals, scores, predictions, namelist


def NetPrediction2(dataloader, model, Cls, criterion):
    reals = np.array([])
    scores = np.empty([0, Cls])
    predictions = np.array([])
    namelist = np.array([])
    losses = 0
    model.eval()
    with torch.no_grad():
        for i, (img, target, name) in tqdm(enumerate(dataloader)):
            out = model(img.cuda())

            prob = F.softmax(out, 1, _stacklevel=5)
            pred = torch.argmax(prob, dim=1)

            reals = np.concatenate((reals, target), axis=0)
            predictions = np.concatenate((predictions, pred.cpu().numpy()), axis=0)
            scores = np.concatenate((scores, prob.cpu().numpy()), axis=0)
            scores = np.reshape(scores, (-1, Cls))
            namelist = np.concatenate((namelist, name), axis=0)

            target = target.cuda().long()
            loss = criterion(out, target)
            losses += loss.item()

            #print('Iteration {:3d} loss {:.6f}'.format(i + 1, loss.item()))
        avgloss = losses / (i + 1)
        print('验证集loss:', avgloss)
    return reals, scores, predictions, namelist, avgloss

def ProbBoxPlot(scores, reals, figname = None):
    plt.figure(figsize=(5, 5))
    plt.boxplot([scores[reals == 0], scores[reals == 1]], sym='',
                positions=[0, 1], widths=0.5, patch_artist=True,
                labels=['Not mutated', 'mutated'], autorange=True, meanline=True)

    _, pval_ttest = stats.ttest_ind(scores[reals == 0], scores[reals == 1])
    print('Prob TTest: P-value : ', pval_ttest)
    _, pval_mann = stats.mannwhitneyu(scores[reals == 0], scores[reals == 1], use_continuity=True, alternative=None)
    print('Prob mannwhitneyu: P-value : ', pval_mann)

    if figname is None:
        plt.show()
    else:
        plt.show()
        plt.savefig(figname, dpi=600, quality=95)
    return pval_ttest, pval_mann


def patient_res_fortrain(reals_patch, scores_patch, namelist_patch):
    reals_patient = np.array([])
    scores_patient = np.array([])
    predictions_patient = np.array([])
    namelist_patient = np.array([])
    pid = np.array([name.split('\\')[-1].split('_')[0] for name in namelist_patch])
    u, counts = np.unique(pid, return_counts=True)
    # print('==========Unique patient ID in name==========\n', u)
    # print('==========Samples count for each id==========\n', counts)
    for id in u:
        sid_label = reals_patch[pid == id]
        sid_score = scores_patch[pid == id]
        reals_patient = np.append(reals_patient, np.mean(sid_label))
        scores_patient = np.append(scores_patient, np.mean(sid_score))
        namelist_patient = np.append(namelist_patient, id)

    auc_patient, threshold_YI_patient = RocPlot(reals_patient, scores_patient)
    for i in range(len(scores_patient)):
        if scores_patient[i] >= threshold_YI_patient:
            predictions_patient = np.append(predictions_patient, np.array([1]))
        else:
            predictions_patient = np.append(predictions_patient, np.array([0]))
    return reals_patient, scores_patient, predictions_patient, namelist_patient, auc_patient, threshold_YI_patient


def patient_res_forval(reals_patch, scores_patch, namelist_patch, threshold_fromtrain):
    reals_patient = np.array([])
    scores_patient = np.array([])
    predictions_patient = np.array([])
    namelist_patient = np.array([])
    pid = np.array([name.split('\\')[-1].split('_')[0] for name in namelist_patch])
    u, counts = np.unique(pid, return_counts=True)
    # print('==========Unique patient ID in name==========\n', u)
    # print('==========Samples count for each id==========\n', counts)
    for id in u:
        sid_label = reals_patch[pid == id]
        sid_score = scores_patch[pid == id]
        reals_patient = np.append(reals_patient, np.mean(sid_label))
        scores_patient = np.append(scores_patient, np.mean(sid_score))
        namelist_patient = np.append(namelist_patient, id)

    auc_patient, threshold_YI_patient_val = RocPlot(reals_patient, scores_patient)

    for i in range(len(scores_patient)):
        if scores_patient[i] >= threshold_fromtrain:
            predictions_patient = np.append(predictions_patient, np.array([1]))
        else:
            predictions_patient = np.append(predictions_patient, np.array([0]))
    return reals_patient, scores_patient, predictions_patient, namelist_patient, auc_patient, threshold_fromtrain, threshold_YI_patient_val


def patient_res_m1(reals_patch, predictions_patch, namelist_patch, Cls):
    reals_patient = np.array([])
    scores_patient = np.empty([0, Cls])
    predictions_patient = np.array([])
    namelist_patient = np.array([])
    pid = np.array([name.split('\\')[-1].split('_')[0] for name in namelist_patch])
    u, counts = np.unique(pid, return_counts=True)
    for id in u:
        sid_label = reals_patch[pid == id]
        sid_prediction = predictions_patch[pid == id]
        #sid_score = scores_patch[pid == id]

        reals_patient = np.append(reals_patient, np.mean(sid_label))
        predictions_patient = np.append(predictions_patient, stats.mode(sid_prediction)[0][0]) ##patch层面预测结果0和1哪个多就认为是哪个

        sid_score = np.array([sum(sid_prediction == 0) / len(sid_prediction), sum(sid_prediction == 1) / len(sid_prediction)])
        scores_patient = np.append(scores_patient, sid_score)
        scores_patient = np.reshape(scores_patient, (-1, Cls))
        namelist_patient = np.append(namelist_patient, id)

    return reals_patient, scores_patient, predictions_patient, namelist_patient



def patient_res_m2(reals_patch, scores_patch, namelist_patch, Cls):
    reals_patient = np.array([])
    scores_patient = np.empty([0, Cls])
    predictions_patient = np.array([])
    namelist_patient = np.array([])
    pid = np.array([name.split('\\')[-1].split('_')[0] for name in namelist_patch])
    u, counts = np.unique(pid, return_counts=True)
    for id in u:
        sid_label = reals_patch[pid == id]
        sid_score = scores_patch[pid == id, :]
        for i in range(sid_score.shape[0]):
            sid_score[i,np.where(sid_score[i,:] != np.max(sid_score[i,:]))] = 0
        sid_score_mean = sid_score.mean(axis=0)
        reals_patient = np.append(reals_patient, np.mean(sid_label))
        scores_patient = np.append(scores_patient, sid_score_mean) ##axis=0是按照列求和
        scores_patient = np.reshape(scores_patient, (-1, Cls))
        predictions_patient = np.append(predictions_patient, np.where(sid_score_mean == np.max(sid_score_mean)))
        namelist_patient = np.append(namelist_patient, id)

    return reals_patient, scores_patient, predictions_patient, namelist_patient


def patient_res_m3(reals_patch, scores_patch, namelist_patch, Cls):
    reals_patient = np.array([])
    scores_patient = np.empty([0, Cls])
    predictions_patient = np.array([])
    namelist_patient = np.array([])
    pid = np.array([name.split('\\')[-1].split('_')[0] for name in namelist_patch])

    u, counts = np.unique(pid, return_counts=True)
    for id in u:
        sid_label = reals_patch[pid == id]
        sid_score = scores_patch[pid == id, :]
        sid_score_mean = sid_score.mean(axis=0)

        reals_patient = np.append(reals_patient, np.mean(sid_label))
        scores_patient = np.append(scores_patient, sid_score_mean) ##axis=0是按照列求和
        scores_patient = np.reshape(scores_patient, (-1, Cls))
        predictions_patient = np.append(predictions_patient, np.where(sid_score_mean == np.max(sid_score_mean)))
        namelist_patient = np.append(namelist_patient, id)

    return reals_patient, scores_patient, predictions_patient, namelist_patient



def EvalMetrics(real, prediction):
    TP = ((real == 1) & (prediction == 1)).sum()  # label 1 is positive
    FN = ((real == 1) & (prediction == 0)).sum()
    TN = ((real == 0) & (prediction == 0)).sum()  # label 0 is negtive
    FP = ((real == 0) & (prediction == 1)).sum()
    print('==============================')
    print('          |  predict          ')
    print('          |  Postive  Negtive ')
    print('==============================')
    print('  Postive | ', TP, '  ', FN, '     = ', TP + FN)
    print('  Negtive | ', FP, '  ', TN, '     = ', TN + FP)
    print('==============================')
    res = {}
    res['Accuracy'] = (TP + TN) / (TP + TN + FP + FN)
    res['Specificity'] = TN / (TN + FP)
    res['Recall'] = TP / (TP + FN)
    res['Precision'] = TP / (TP + FP)
    res['F1Score'] = (2 * res['Recall'] * res['Precision']) / (res['Recall'] + res['Precision'])
    #    return [Accuracy, Specificity, Recall, Precision, F1Score]
    return res

def othermetrics(real, prediction):

    res = {}
    res['Accuracy'] = metrics.accuracy_score(real, prediction)
    res['balanceAcc'] = metrics.balanced_accuracy_score(real, prediction)
    res['Recall'] = metrics.recall_score(real, prediction, average='weighted')
    res['Precision'] = metrics.precision_score(real, prediction, average='weighted')
    res['F1Score'] = metrics.f1_score(real, prediction, average='weighted')
    res['cohen_kappa_score'] = metrics.cohen_kappa_score(real, prediction)
    print(res['cohen_kappa_score'])

    print(metrics.classification_report(real, prediction))
    return res


def EvalMetricsV2(real, prediction, confMat=True, savename=None):
    # targetName = ['tumor', 'stroma', 'immune', 'duct', 'necrosis', 'vessel']
    targetName = ['BLIS', 'IM', 'LAR', 'MES']
    res = {}
    res['Accuracy'] = metrics.accuracy_score(real, prediction)
    res['balanceAcc'] = metrics.balanced_accuracy_score(real, prediction)
    res['Recall'] = metrics.recall_score(real, prediction, average='weighted')
    res['Precision'] = metrics.precision_score(real, prediction, average='weighted')
    res['F1Score'] = metrics.f1_score(real, prediction, average='weighted')

    sns.set()

    if confMat:
        res['confMatrix'] = metrics.confusion_matrix(real, prediction,
                                                     normalize='true')  # normalize : {'true', 'pred', 'all'}, default=None
        #        Normalizes confusion matrix over the true (rows), predicted (columns)
        #        conditions or all the population. If None, confusion matrix will not be normalized.
        ax = sns.heatmap(res['confMatrix'], annot=True, cmap='jet', square=True, fmt='.2f',  # 'd' /'.2f'
                         xticklabels=targetName, yticklabels=targetName)  # heat map
    else:
        ax = sns.heatmap(metrics.confusion_matrix(real, prediction, normalize='true'),
                         annot=True, cmap='jet', square=True, fmt='.2f',  # 'd' /'.2f'
                         xticklabels=targetName, yticklabels=targetName)  # heat map
    ax.set_title('confusion matrix')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    if savename is None:
        plt.show()
    else:
        plt.savefig(savename, dpi=600, quality=95)

    print(metrics.classification_report(real, prediction, target_names=targetName))
    return res


def plt_ring(percentage_list, color_list, fig_name):
    plt.figure(figsize=(5, 5))
    plt.pie(percentage_list,
            radius=1,
            pctdistance=0.85,
            wedgeprops=dict(width=0.5, edgecolor='w'),
            colors=color_list,
            startangle=90,
            textprops={'color': 'w'},
            )
    plt.savefig(fig_name, dpi=600, quality=95)
    plt.close('all')


def GetTrainArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('-configs', '--configs', default='../results/HRD_resnet18_0.1_fromScratch_setting.txt',
                        type=str,
                        required=False, help='save config files.')

    parser.add_argument('-E', '--epoches', default=300, type=int, required=False, help='Epoch, default is 300.')
    parser.add_argument('-B', '--batch_size', default=256, type=int, required=False, help='batch size, default is 256.')
    parser.add_argument('-LR', '--initLR', default=0.005, type=float, required=False, help='init lr, default is 0.003.')
    parser.add_argument('-Wg', '--weights', default=None, type=list, required=False, help='weights for CEloss.')
    # weights for loss; or weights = None

    parser.add_argument('-trainpath', '--trainpath', default='../data/0.1_train_HRD.txt', type=str,
                        required=False, help='trainpath, default is. ')
    parser.add_argument('-validpath', '--validpath', default='../data/0.1_val_HRD.txt', type=str,
                        required=False, help='valpath, default is.')
    parser.add_argument('-preroot', '--preroot', default='/home/cyyan/projects/tnbc/data/Wpatch/', type=str,
                        required=False, help='preroot, default is /home/cyyan/projects/tnbc/data/Wpatch/')
    parser.add_argument('-norm', '--norm',
                        default={'normMean': [0.728, 0.4958, 0.7047], 'normStd': [0.1513, 0.1666, 0.1121]},
                        type=dict, required=False, help='normMean and Std for data normalization.')

    parser.add_argument('-sn', '--savename', default='../models', type=str, required=False,
                        help='savename for model saving, default is ../models.')
    parser.add_argument('-logdir', '--logdir', default='../results/logs/fromScratch', type=str, required=False,
                        help='logdir for tensorboardX, default is ../results/logs.')

    parser.add_argument('-net', '--net', default='resnet18', type=str,
                        required=False, help='network from torchvision for classification, default is resnet18')
    parser.add_argument('-restore', '--restore', default='', type=str, required=False,
                        help='Model path restoring for testing, if none, just \'\', no default.')
    parser.add_argument('-pretrained', '--pretrained',
                        default='/home/cyyan/.cache/torch/checkpoints/resnet18-5c106cde.pth',
                        type=str, required=False,
                        help='Model path pretrained for training, if none, just \'\', no default.')
    # '/home/cyyan/.cache/torch/checkpoints/alexnet-owt-4df8aa71.pth'
    # '/home/cyyan/.cache/torch/checkpoints/resnet18-5c106cde.pth'
    parser.add_argument('-loss', '--loss', default='CrossEntropyLoss', type=str,
                        required=False, help='loss function for classification, default is CrossEntropyLoss')

    parser.add_argument('-C', '--nCls', default=2, type=int, required=False, help='num of Class, here is 2.')
    parser.add_argument('-W', '--nWorker', default=8, type=int, required=False,
                        help='Num worker for dataloader, default is 8.')
    parser.add_argument('-mo', '--momentum', default=0.9, type=float, required=False, help='momentum, default is 0.9.')
    parser.add_argument('-de', '--decay', default=1e-5, type=float, required=False, help='decay, default is 1e-5.')
    parser.add_argument('-S', '--seed', default=2020, type=int, required=False, help='random seed, default 2020.')
    parser.add_argument('-G', '--gpu', default='0', type=str, required=False, help='one or multi gpus, default is 0.')

    args = parser.parse_args()
    with open(args.configs, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # with open('setting.txt', 'r') as f:
    #    args.__dict__ = json.load(f)
    return args


def GetParser():
    parser = argparse.ArgumentParser(description='Binary Classification by PyTorch')
    parser.add_argument('--config', type=str, default='myParams.yaml', help='config file')
    parser.add_argument('opts', help='see Params.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None, "Please provide config file for myParams."
    cfg = myConfig.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = myConfig.merge_cfg_from_list(cfg, args.opts)
    with open(cfg.configs, 'w') as f:
        json.dump(cfg, f, indent=2)
    return cfg


if __name__ == "__main__":
    pass
