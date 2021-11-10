import os
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
import myInception_v3
from myDataReader import ClsDataset
from myUtils import save_temp_excel, GetParser, ProbBoxPlot, NetPrediction, EvalMetrics, EvalMetricsV2, patient_res_forval, RocPlot, patient_res_m3

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

    testset = ClsDataset(args.testpath, args.test_root, preprocess)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.nWorker)
    print(args.testpath)

    if args.net == 'inception_v3':
        net = getattr(myInception_v3, args.net)(pretrained=False, num_classes=args.nCls)
    else:
        net = getattr(models, args.net)(pretrained=False, num_classes=args.nCls)

    if len(args.gpu) > 1:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = net.cuda()

    rootpath = os.path.join(args.resultpath, args.task, args.taskID)
    net.load_state_dict(
        torch.load(os.path.join(rootpath, 'models', args.restore)))  # load the finetune weight parameters
    print('####################Loading model...', os.path.join(rootpath, 'models', args.restore))

    savepath = os.path.join(rootpath, args.restore, 'TEST')
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    reals_patch_test, scores_patch_test, predictions_patch_test, namelist_patch_test = NetPrediction(testloader, net, args.nCls)


    if args.nCls == 2:
        result_patch = EvalMetrics(reals_patch_test, predictions_patch_test)
        auc_patch_test, threshold_YI_patch_test = RocPlot(reals_patch_test, scores_patch_test[:, 1])
        print("testing set patch-level AUC:", auc_patch_test,
            "testing set patch-level threshold：", threshold_YI_patch_test)
    elif args.nCls == 4:
        result_patch = EvalMetricsV2(reals_patch_test, predictions_patch_test)

    for key in result_patch:
        print(key, ': ', result_patch[key])

    ProbBoxPlot(scores_patch_test[:, 1], reals_patch_test)

    savename_patch = os.path.join(savepath, 'patchTEST.npz')
    save_temp_excel(namelist_patch_test, scores_patch_test[:, 1], predictions_patch_test, reals_patch_test,
                    savepath, args.nCls, 'patch', 'TEST')
    np.savez(savename_patch, key_real=reals_patch_test, key_score=scores_patch_test, key_binpred=predictions_patch_test,
             key_namelist=namelist_patch_test)

    reals_patient_test, scores_patient_test, predictions_patient_test, namelist_patient_test = patient_res_m3(
        reals_patch_test, scores_patch_test, namelist_patch_test, args.nCls)


    ProbBoxPlot(scores_patient_test[:, 1], reals_patient_test)

    if args.nCls == 2:
        result_patient = EvalMetrics(reals_patient_test, predictions_patient_test)
        auc_patient_test, threshold_YI_patient_test = RocPlot(reals_patient_test, scores_patient_test[:, 1])
        print("testing set patient-level AUC:", auc_patient_test,
            "testing set patient-level threshold：", threshold_YI_patient_test)
    elif args.nCls == 4:
        result_patient = EvalMetricsV2(reals_patient_test, predictions_patient_test)

    for key in result_patient:
        print(key, ': ', result_patient[key])

    save_temp_excel(namelist_patient_test, scores_patient_test[:, 1], predictions_patient_test, reals_patient_test,
                                     savepath, args.nCls, 'patient', 'TEST')
    savename_patient = os.path.join(savepath, 'patientTEST.npz')
    np.savez(savename_patient, key_real=reals_patient_test, key_score=scores_patient_test, key_binpred=predictions_patient_test,
             key_namelist=namelist_patient_test)

if __name__ == '__main__':
    arg = GetParser()
    main(arg)
