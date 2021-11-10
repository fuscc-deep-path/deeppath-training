import os
import numpy as np
import random
import myInception_v3
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from myDataReader import ClsDataset
from myUtils import save_temp_excel, GetParser, ProbBoxPlot, NetPrediction, EvalMetrics, EvalMetricsV2, RocPlot, patient_res_m3, patient_res_m2, patient_res_m3

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(), #operated on original image, rewrite on previous transform.
        transforms.Normalize(args.norm['normMean'], args.norm['normStd'])])

    valset = ClsDataset(args.validpath, args.traval_root, preprocess)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.nWorker)
    print(args.validpath)

    if args.net == 'inception_v3':
        net = getattr(myInception_v3, args.net)(pretrained=False, num_classes=args.nCls)
    else:
        net = getattr(models, args.net)(pretrained=False, num_classes=args.nCls)

    if len(args.gpu) > 1:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = net.cuda()

    rootpath = os.path.join(args.resultpath, args.task, args.taskID)
    net.load_state_dict(torch.load(os.path.join(rootpath, 'models', args.restore))) # load the finetune weight parameters

    savepath = os.path.join(rootpath, args.restore, 'VAL')
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    reals_patch_val, scores_patch_val, predictions_patch_val, namelist_patch_val = NetPrediction(valloader, net, args.nCls)

    if args.nCls == 2:
        result_patch = EvalMetrics(reals_patch_val, predictions_patch_val)
        auc_patch_val, threshold_YI_patch_val = RocPlot(reals_patch_val, scores_patch_val[:, 1])
        print("validation set patch-level AUC:", auc_patch_val,
            "validation set patch-level threshold：", threshold_YI_patch_val)
    elif args.nCls == 4:
        result_patch = EvalMetricsV2(reals_patch_val, predictions_patch_val)

    for key in result_patch:
        print(key, ': ', result_patch[key])

    ProbBoxPlot(scores_patch_val[:, 1], reals_patch_val)

    savename_patch = os.path.join(savepath, 'patchVAL.npz')
    save_temp_excel(namelist_patch_val, scores_patch_val[:, 1], predictions_patch_val, reals_patch_val,
                    savepath, args.nCls, 'patch', 'VAL')
    np.savez(savename_patch, key_real=reals_patch_val, key_score=scores_patch_val, key_binpred=predictions_patch_val,
             key_namelist=namelist_patch_val)

    reals_patient_val, scores_patient_val, predictions_patient_val, namelist_patient_val = patient_res_m3(
        reals_patch_val, scores_patch_val, namelist_patch_val, args.nCls)

    ProbBoxPlot(scores_patient_val[:, 1], reals_patient_val)

    if args.nCls == 2:
        result_patient = EvalMetrics(reals_patient_val, predictions_patient_val)
        auc_patient_val, threshold_YI_patient_val = RocPlot(reals_patient_val, scores_patient_val[:, 1])
        print("validation set patient-level AUC:", auc_patient_val,
            "validation set patient-level threshold：", threshold_YI_patient_val)
    elif args.nCls == 4:
        result_patient = EvalMetricsV2(reals_patient_val, predictions_patient_val)

    for key in result_patient:
        print(key, ': ', result_patient[key])

    save_temp_excel(namelist_patient_val, scores_patient_val[:, 1], predictions_patient_val, reals_patient_val,
                                     savepath, args.nCls, 'patient', 'VAL')
    savename_patient = os.path.join(savepath, 'patientVAL.npz')
    np.savez(savename_patient, key_real=reals_patient_val, key_score=scores_patient_val, key_binpred=predictions_patient_val,
             key_namelist=namelist_patient_val)


if __name__ == '__main__':
    arg = GetParser()
    main(arg)
