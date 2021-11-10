import os
import numpy as np

def getratio(info, label_num):
    trainsample_num_list = []
    for i in range(0, label_num, 1):
        trainsample_num = len([trainsample for trainsample in info if trainsample[0] == 'train' and trainsample[2] == str(i)])
        trainsample_num_list.append(trainsample_num)
    min_sample = min(trainsample_num_list)
    ratiolist = [a/min_sample for a in trainsample_num_list]
    return ratiolist


def down_sampling(patch_num_base, label, ratiolist):
    ratio = ratiolist[eval(label)]
    ends = int(patch_num_base / ratio) if int(patch_num_base / ratio) < len(tumor_patchlist) else len(tumor_patchlist)
    return ends


def get_task_ID(fold, patch_num_base, scores_cutoff = None):
    if scores_cutoff is None:
        task_ID = fold + '_alltumor' + '_patch' + str(patch_num_base)
    else:
        task_ID = fold + '_cutoff' + str(scores_cutoff) + '_patch' + str(patch_num_base)
    print(task_ID)
    return task_ID

np.random.seed(2020)
gene_wetest = 'PIK3CA'
fold_num = 'fold2' #fold0 fold1 fold2
scores_cutoff = None
patch_num_base_train = 500
patch_num_base_valtest = 250

netdata_dir = 'F:\\TNBC_DL\\DATA\\NET_DATA\\TRY_TEMP_NORM_NEW\\TrValTe_PATIENT_LABEL\\'

task = gene_wetest + '_mutation'
task_ID = get_task_ID(fold_num, patch_num_base_train, scores_cutoff)
netdatapath = os.path.join(netdata_dir, task)

f = open(netdatapath + '\\split_Patient_TrValTe_' + task + '_' +fold_num + '.txt', 'r')

info = []
for line in f:
    line = line.rstrip()
    words = line.split(' ')
    info.append((words[0], words[1], words[2]))

ratiolist = getratio(info, 2)
print(ratiolist)

write_dir = 'F:\\TNBC_DL\\DATA\\NET_DATA\\TRY_TEMP_NORM_NEW\\TrValTe_PATCH_LABEL\\'

writepath = os.path.join(write_dir, task)
if not os.path.exists(writepath):
    os.makedirs(writepath)

train = open(os.path.join(writepath, task_ID + '_TRAIN.txt'), 'w')
val = open(os.path.join(writepath, task_ID + '_VAL.txt'), 'w')
test = open(os.path.join(writepath, task_ID + '_TEST.txt'), 'w')

npzpath_traval = 'F:\\TNBC_DL\\DATA\\IMAGE_DATA\\FUSCC_TNBC_IMAGES\\FUSCC_NORM_FEATS_NEW\\'
npzpath_test = 'F:\\TNBC_DL\\DATA\\IMAGE_DATA\\TCGA_TNBC_IMAGES\\TCGA_NORM_FEATS_NEW\\'


for typ, slide, label in info:

    files = os.path.join(npzpath_traval, slide, slide + '.npz') if typ in ['train', 'val'] else os.path.join(npzpath_test, slide, slide + '.npz')
    if not os.path.exists(files):
        continue
    data = np.load(files)
    scores, bins, namelist = data['score'], data['bin'], data['namelist']
    namelist_cut = np.array([name.split('\\')[-1].split('_')[0] +'\\' + name.split('\\')[-1] for name in namelist])

    if scores_cutoff is None:
        tumor_patchlist = namelist_cut[bins == 0]
    else:
        tumor_patchlist = namelist_cut[scores[:, 0] > scores_cutoff]

    if len(tumor_patchlist) < 20:
        print('low tumor patch number slide', slide, 'tumor patches number:', len(tumor_patchlist))
        continue
    else:
        print(slide, ' tumor patches number: ', len(tumor_patchlist))
    np.random.shuffle(tumor_patchlist)

    if typ in ['train']:
        ends = down_sampling(patch_num_base_train, label, ratiolist)
    elif typ in ['val', 'test']:
        ends = patch_num_base_valtest if patch_num_base_valtest < len(tumor_patchlist) else len(tumor_patchlist)

    tumor_patchlist_sampled = tumor_patchlist[0:ends]

    for s in tumor_patchlist_sampled:
        name = os.path.join(s + '.png')
        eval(typ).write(name + ' ' + label + '\n')


f.close()
train.close()
val.close()