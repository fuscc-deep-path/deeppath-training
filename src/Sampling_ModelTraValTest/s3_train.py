import os
import time
import numpy as np
import random
import torch
import myLoss
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models
from tensorboardX import SummaryWriter
import myInception_v3
import myTransforms
from myDataReader import ClsDataset
from myUtils import GetParser, Train, NetPrediction, NetPrediction2, EvalMetrics, RocPlot, patient_res_m3

def main(args):
    start = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    preprocess = myTransforms.Compose([
        myTransforms.Resize((256, 256)),
        # myTransforms.RandomChoice([myTransforms.RandomHorizontalFlip(p=0.5),
        #                            myTransforms.RandomVerticalFlip(p=0.5),
        #                            myTransforms.AutoRandomRotation()]),
        # myTransforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), hue=0.3),
        myTransforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0.3),

        #myTransforms.HEDJitter(theta=0.05),
        #myTransforms.RandomAffine(degrees=0, scale=[0.9, 1.1]),
        # myTransforms.RandomErasing(),
        # myTransforms.RandomCrop(256, 16),
        myTransforms.ToTensor(),  # operated on original image, rewrite on previous transform.
        myTransforms.Normalize(args.norm['normMean'], args.norm['normStd'])])

    valprocess = myTransforms.Compose([myTransforms.Resize((256, 256)),
                                       myTransforms.ToTensor(),
                                       myTransforms.Normalize(args.norm['normMean'], args.norm['normStd'])])

    print('####################Loading dataset...')
    trainset = ClsDataset(args.trainpath, args.traval_root, preprocess)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.nWorker)
    valset = ClsDataset(args.validpath, args.traval_root, valprocess)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.nWorker)

    if args.net == 'inception_v3':
        net = getattr(myInception_v3, args.net)(pretrained=False, num_classes=args.nCls)
    else:
        net = getattr(models, args.net)(pretrained=False, num_classes=args.nCls)
    print('network defining')

    if len(args.gpu) > 1:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = net.cuda()

    if args.restore:
        net.load_state_dict(torch.load(args.restore))  # load the finetune weight parameters
        print('####################Loading model...', args.restore)
    elif args.pretrained:
        net_state_dict = net.state_dict()  # get the new network dict
        pretrained_dict = torch.load(args.pretrained)  # load the pretrained model
        pretrained_dict_new = {k: v for k, v in pretrained_dict.items() if k in net_state_dict and net_state_dict[
            k].size() == v.size()}  # check the same key in dict.items
        net_state_dict.update(pretrained_dict_new)  # update the new network dict by new dict in pretrained
        net.load_state_dict(net_state_dict)  # load the finetune weight parameters
        print('####################Loading pretrained model from torch cache checkpoints...')

    print('####################Loading criterion and optimizer...')
    weights = args.weights if args.weights is None else torch.tensor(args.weights).cuda()
    if args.loss == 'FocalLoss':
        criterion = getattr(myLoss, args.loss)(weight=weights).cuda()
    else:
        criterion = getattr(nn, args.loss)(weight=weights).cuda()
    print(args.loss)


    for i, para in enumerate(net.parameters()):  ##resnet18:45  resnet34:87  alexnet:8   inception_v3:  shufflenet:165
        if i < args.froz_layer:
            para.requires_grad = False
        else:
            para.requires_grad = True
    for name, para in net.named_parameters():
        print(name, para.requires_grad)

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.initLR, momentum=args.momentum, weight_decay=args.decay)
    elif args.optimizer == 'ADAM':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.initLR, weight_decay=args.decay)

    summary_path = os.path.join(args.resultpath, args.task, args.taskID, 'summary4tensorboard')
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    trainwriter = SummaryWriter(log_dir='{}/{}'.format(summary_path, 'train'))
    valwriter = SummaryWriter(log_dir='{}/{}'.format(summary_path, 'val'))

    iterations = int(trainset.__len__() / args.batch_size)

    for epoch in range(args.epoches):
        # AdjustLR(optimizer, epoch, args.epoches, args.initLR, power=args.power)
        # print('Current LR:', optimizer.param_groups[0]['lr'])

        net, avgloss = Train(trainloader, net, optimizer, criterion, epoch, iterations, args.b)

        model_savepath = os.path.join(args.resultpath, args.task, args.taskID, 'models')
        if not os.path.exists(model_savepath):
            os.makedirs(model_savepath)
        torch.save(net.state_dict(), os.path.join(model_savepath, str(epoch) + '.pkl'))
        trainwriter.add_scalar('Training classification loss', avgloss, epoch)
        print('Model has been saved!')

        reals_patch_train, scores_patch_train, predictions_patch_train, namelist_patch_train = NetPrediction(trainloader, net, args.nCls)
        auc_patch_train, threshold_YI_patch_train = RocPlot(reals_patch_train, scores_patch_train[:, 1])
        print("training set patch-level AUC:", auc_patch_train,
              "training set patch-level threshold：", threshold_YI_patch_train)
        reals_patient_train, scores_patient_train, predictions_patient_train, namelist_patient_train = patient_res_m3(reals_patch_train, scores_patch_train, namelist_patch_train, args.nCls)
        auc_patient_train, threshold_YI_patient_train = RocPlot(reals_patient_train, scores_patient_train[:, 1])

        print("training set patient-level AUC:", auc_patient_train,
              "training set patient-level thresholdYI:", threshold_YI_patient_train)
        result = EvalMetrics(reals_patient_train, predictions_patient_train)
        for key in result:
            print(key, ': ', result[key])
            trainwriter.add_scalar(key, result[key], epoch)
        trainwriter.add_scalar('Train AUC patch', auc_patch_train, epoch)

        reals_patch_val, scores_patch_val, predictions_patch_val, namelist_patch_val, val_loss = NetPrediction2(valloader, net, args.nCls, criterion)
        valwriter.add_scalar('Validation classification loss', val_loss, epoch)

        auc_patch_val, threshold_YI_patch = RocPlot(reals_patch_val, scores_patch_val[:,1])
        print("validation set patch-level AUC:", auc_patch_val,
              "validation set patch-level thresholdYI：", threshold_YI_patch)
        reals_patient_val, scores_patient_val, predictions_patient_val, namelist_patient_val = patient_res_m3(reals_patch_val, scores_patch_val, namelist_patch_val, args.nCls)

        auc_patient_val, threshold_YI_patient_val = RocPlot(reals_patient_val, scores_patient_val[:, 1])


        print("validation set patient-level AUC:", auc_patient_val,
              "validation set patient-level thresholdYI val:", threshold_YI_patient_val)
        result = EvalMetrics(reals_patient_val, predictions_patient_val)
        for key in result:
            print(key, ': ', result[key])
            valwriter.add_scalar(key, result[key], epoch)
        valwriter.add_scalar('Val AUC patch', auc_patch_val, epoch)
        valwriter.add_scalar('Val AUC patient', auc_patient_val, epoch)
        print('Time consuming (sec): ', time.time() - start)

    trainwriter.close()
    valwriter.close()


if __name__ == '__main__':
    arg = GetParser()
    main(arg)
