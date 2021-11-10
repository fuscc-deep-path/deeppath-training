import os
import numpy as np
import pandas as pd

def get_PID(clinical_file, mutation_file, symbol):
    clinical_data = pd.read_excel(clinical_file)
    with_WES = np.array([clinical_data['Project_ID'][clinical_data['Exome_Seqencing'] == 'YES']])

    mutation_data = pd.read_excel(mutation_file)
    mutated_ID = np.unique(mutation_data['Tumor_Sample_Barcode'][mutation_data['Hugo_Symbol'] == symbol])

    not_mutated_ID = np.setdiff1d(with_WES, mutated_ID)
    print(len(not_mutated_ID), len(mutated_ID))

    return not_mutated_ID, mutated_ID

def get_PID_TCGA(clinical_file, mutation_file, symbol):
    clinical_data = pd.read_excel(clinical_file)
    with_WES = np.array([clinical_data['SAMPLE_ID'][clinical_data['cases_sequenced'] == 'YES']])

    mutation_data = pd.read_table(mutation_file)
    mutated_ID = np.unique(mutation_data['Tumor_Sample_Barcode'][mutation_data['Hugo_Symbol'] == symbol])

    not_mutated_ID = np.setdiff1d(with_WES, mutated_ID)
    print(len(not_mutated_ID), len(mutated_ID))

    return not_mutated_ID, mutated_ID


def fidwrite_fold(ID_label, label_seq, fold, task):
    for i in range(fold):

        savetxtfile = os.path.join(save_path, 'split_Patient_TrValTe_' + task + '_fold' +str(i) + '.txt')
        if label_seq == 0:
            fid = open(savetxtfile, 'w')
        else:
            fid = open(savetxtfile, 'a')

        ID_label_train, ID_label_val = get_kfold_data(fold, i, ID_label)
        for ID in ID_label_train:
            content = "train " + ID + " " + str(label_seq)
            fid.write(content + "\n")
        for ID in ID_label_val:
            content = "val " + ID + " " + str(label_seq)
            fid.write(content + "\n")
        fid.close()


def get_kfold_data(k, i, pID):

    fold_size = round(len(pID) / k)

    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        pID_val= pID[val_start:val_end]
        pID_train = np.concatenate((pID[0:val_start], pID[val_end:]), axis = 0)
    else:
        pID_val = pID[val_start:]
        pID_train = pID[0:val_start]

    return pID_train, pID_val


def fid_write_test(ID_label, label_seq, fold, task):
    for i in range(fold):

        savetxtfile = os.path.join(save_path, 'split_Patient_TrValTe_' + task + '_fold' + str(i) + '.txt')
        fid = open(savetxtfile, 'a')
        for ID in ID_label:

            content = "test " + ID + " " + str(label_seq)
            fid.write(content + "\n")
        fid.close()



np.random.seed(2020)
## PTEN RB1  PIK3R1 KMT2C PIK3CA TP53 FAT3 KMT2D
gene_wetest = 'PIK3CA'

clinical_file = 'F:\\TNBC_DL\\DATA\\MEDICAL_DATA\\FUSCC_TNBC_DATA\\FUSCCTNBC_MergedClinicalOmicsData_V15_190209.xlsx'
mutation_file = 'F:\\TNBC_DL\\DATA\\MEDICAL_DATA\\FUSCC_TNBC_DATA\\FUSCCTNBC_Mutations_MultiCaller_V15_ExcludeIGR_190209.xlsx'

PID1, PID2 = get_PID(clinical_file, mutation_file, gene_wetest)

patchpath = 'F:\\TNBC_DL\\DATA\\IMAGE_DATA\\FUSCC_TNBC_IMAGES\\FUSCC_NORM_PATCH\\'

foldernames = np.array([os.listdir(patchpath)])

ID_label1 = foldernames[np.isin(foldernames, PID1)]
np.random.shuffle(ID_label1)
ID_label2 = foldernames[np.isin(foldernames, PID2)]
np.random.shuffle(ID_label2)

save_dir = 'F:\\TNBC_DL\\DATA\\NET_DATA\\TRY_TEMP_NORM_NEW\\TrValTe_PATIENT_LABEL\\'

task = gene_wetest + '_mutation'
save_path = os.path.join(save_dir, task)
if not os.path.exists(save_path):
    os.makedirs(save_path)

fidwrite_fold(ID_label=ID_label1, label_seq=0, fold=3, task=task)
fidwrite_fold(ID_label=ID_label2, label_seq=1, fold=3, task=task)

clinical_file_TCGA = 'F:\\TNBC_DL\\DATA\\MEDICAL_DATA\\TCGA_TNBC_DATA\\brca_tcga\\data_bcr_clinical_data_sample.xlsx'
mutation_file_TCGA = 'F:\\TNBC_DL\\DATA\\MEDICAL_DATA\\TCGA_TNBC_DATA\\brca_tcga\\data_mutations_mskcc.txt'
PID1, PID2 = get_PID_TCGA(clinical_file_TCGA, mutation_file_TCGA, gene_wetest)

patchpath = 'F:\\TNBC_DL\\DATA\\IMAGE_DATA\\TCGA_TNBC_IMAGES\\TCGA_NORM_PATCH\\'

foldernames = np.array([a for a in os.listdir(patchpath)])

ID_label1 = foldernames[np.isin(foldernames, [a + 'Z-00-DX1' for a in PID1])]
np.random.shuffle(ID_label1)
ID_label2 = foldernames[np.isin(foldernames, [a + 'Z-00-DX1' for a in PID2])]
np.random.shuffle(ID_label2)

fid_write_test(ID_label=ID_label1, label_seq=0, fold=3, task=task)
fid_write_test(ID_label=ID_label2, label_seq=1, fold=3, task=task)
