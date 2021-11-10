import os
import time
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
import csv
import torch
from torch.utils import data
import torch.nn.functional as F
from torchvision import models, transforms

class TumorDetPatchReader_bkg(data.Dataset):

    def __init__(self, folder, format):
        self.fg_patchs = self.check_bkg(folder, format)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.8201, 0.5207, 0.7189], [0.1526, 0.1542, 0.1183])])

    def __getitem__(self, index):
        name = self.fg_patchs[index]
        img = Image.open(name)  # .convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, -1, name.split('/')[-1].split('.')[0]

    def check_bkg(self, folder, format, threshold=210, ratio=0.5):
        fg_patchs = np.array([])
        for name in glob(os.path.join(folder, '*'+format)):
            img = Image.open(name)
            img = img.convert('L')
            img = np.array(img)

            bkg_num = np.sum(img > threshold)
            bkg_cut = len(img.flatten()) * ratio

            if bkg_num < bkg_cut:
                fg_patchs = np.append(fg_patchs, name)

        return fg_patchs

    def __len__(self):
        return len(self.fg_patchs)
# #
class TumorDetPatchReader(data.Dataset):

    def __init__(self, folder, format):
        self.lists = glob(os.path.join(folder, '*'+format))
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.8201, 0.5207, 0.7189], [0.1526, 0.1542, 0.1183])])

    def __getitem__(self, index):
        name = self.lists[index]
        img = Image.open(name)  # .convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, -1, name.split('/')[-1].split('.')[0]

    def __len__(self):
        return len(self.lists)

# load the network
def load_net(modelpath, numclasses=2):
    net = models.resnet18(pretrained=False, num_classes=numclasses)
    net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(modelpath).items()})
    net = net.cuda()   # ## only one cuda can be use
    return net.eval()

# hook the feature extractor
features_blobs = [0]
def hook_feature(module, input, output):
    #    features_blobs.append(output.data.cpu().numpy())
    features_blobs[0] = output.data.cpu().numpy()


# network and image for prediction and extract each patch's feats
# the npz file of tumor detcetion results and the several  .csv files are in the same folder
def net_pred_extract_feats(loader, net, featPth=None, cls=2):
    if featPth is None:
        featPth = ''
    f1 = open(featPth, 'w')
    f2 = open(featPth.replace('Feats.csv', 'Names.csv'), 'w')
    f3 = open(featPth.replace('Feats.csv', 'Scores.csv'), 'w')
    f4 = open(featPth.replace('Feats.csv', 'Predictions.csv'), 'w')

    net._modules.get('avgpool').register_forward_hook(hook_feature)  # get feature maps

    writer1 = csv.writer(f1)
    writer_imgname = csv.writer(f2)
    writer_scores = csv.writer(f3)
    writer_preds = csv.writer(f4)

    score = np.empty([0, cls])
    bin = np.array([])
    namelist = np.array([])

    with torch.no_grad():
        for i, (img, _, name) in tqdm(enumerate(loader)):
            img = img.cuda()
            predProb = F.softmax(net(img), dim=1)
            predBina = torch.argmax(predProb, dim=1)

            writer1.writerow(np.squeeze(features_blobs[0]))
            writer_imgname.writerow(name)
            writer_scores.writerow(predProb.cpu().numpy().squeeze())
            writer_preds.writerow(predBina.cpu().numpy())

            bin = np.concatenate((bin, predBina.cpu().numpy()), axis=0)
            score = np.concatenate((score, predProb.cpu().numpy()), axis=0)
            namelist = np.concatenate((namelist, name), axis=0)

    f1.close()
    f2.close()
    f3.close()
    f4.close()

    return score, bin, namelist


if __name__ == "__main__":
    start = time.time()

    modelpth = '..\\norm_results1\\resnet18_128_CrossEntropyLoss_Adam_0.001_noadj\\models\\NormModel_epoch_105.pkl' #model path

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    net = load_net(modelpth, numclasses=5)

    datapth = 'F:\\TNBC_DL\\DATA\\IMAGE_DATA\\FUSCC_TNBC_IMAGES\\FUSCC_NORM_PATCH\\'
    feats_savepth = 'F:\\TNBC_DL\\DATA\\IMAGE_DATA\\FUSCC_TNBC_IMAGES\\FUSCC_NORM_FEATS_NEW\\'

    if not os.path.exists(feats_savepth):
        os.mkdir(feats_savepth)

    folders = os.listdir(datapth)

    for folder in folders:
        print(folder)
        subfolder = os.path.join(feats_savepth, folder)

        if os.path.exists(subfolder):
            print(folder, 'has been tumor detected')
            continue
        else:
            os.mkdir(subfolder)
            dataset = TumorDetPatchReader_bkg(os.path.join(datapth, folder), format='png')
            loader = data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=8, pin_memory=True)

            scores, bins, namelist = net_pred_extract_feats(loader, net, os.path.join(subfolder, 'Feats.csv'), cls=5)
            print(len(scores), '\n', len(bins), '\n', (namelist))

            np.savez(os.path.join(subfolder, folder+'.npz'), score=scores, bin=bins, namelist=namelist)

    print('Finished! Time consuming (sec): ', time.time() - start)
