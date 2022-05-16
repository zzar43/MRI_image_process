"""
This file is to do the skull strip with FreeSurfer.

Version 2.
Author: Da Li

"""

from copy import copy
# from lib2to3.pytree import BasePattern
import os
from glob import glob
from tqdm import tqdm

import nibabel as nib
import torch
from torch import nn
import numpy as np

CPU_NUM = 4
# set path
# BASE_PATH = os.getcwd()
BASE_PATH = "/Users/lida/Local_Document/ADNI_dataset/UNet_IDP_Stage_0"
ADNI_PATH = BASE_PATH + "/ADNI"
FREESURFER_PATH = BASE_PATH + "/FreeSurfer"
SAVE_PATH = BASE_PATH + "/Dataset"
# get adni mri path
MRI_PATH_LIST = glob(ADNI_PATH + '/**/*.nii', recursive=True)


print('Number of file: ', len(MRI_PATH_LIST))
print(BASE_PATH)
print(ADNI_PATH)
print(FREESURFER_PATH)
print(SAVE_PATH)

def copy_adni_to_freesurfer():
    print("\nCopy ADNI MRI data to FreeSurfer subfolder.")
    for idx in tqdm(range(len(MRI_PATH_LIST))):
        cmd = "cp " + MRI_PATH_LIST[idx] + " " + FREESURFER_PATH
        os.system(cmd)
    print("Copy Done.")

def skull_remove():

    print("\nRemove skull:")
    os.chdir(FREESURFER_PATH)
    print("Change path to: ", FREESURFER_PATH)
    cmd1 = "export SUBJECTS_DIR=" + FREESURFER_PATH + ";"
    # for test
    cmd2 = "ls *.nii | parallel -j" + str(CPU_NUM) + "recon-all -s {.} -i {} > log.txt"
    # for skull remove
    # cmd2 = "ls *.nii | parallel -j " + str(CPU_NUM) + " recon-all -s {.} -i {} -autorecon1 > log_skull_remove.txt"
    print("Start remove skull... Wait for hours...")
    os.system(cmd1 + cmd2)
    os.chdir(BASE_PATH)
    print("Change path to: ", BASE_PATH, "\nDone.")

def convert_mri():
    print("\nConvert MRI from *.mgz to *.nii.")

    os.chdir(FREESURFER_PATH)
    print("Change path to: ", FREESURFER_PATH)
    brainmask_path = glob(FREESURFER_PATH + '/**/brainmask.mgz', recursive=True)

    for mgz_path in brainmask_path:
        current_path = os.path.split(mgz_path)[0]
        filename = os.path.split(mgz_path)[1]
        filename = os.path.splitext(filename)[0]
        cmd = "mri_convert " + mgz_path + " " + current_path + "/" + filename + ".nii > log_mri_convert.txt"
        os.system(cmd)

    os.chdir(BASE_PATH)
    print("Change path to: ", BASE_PATH, "\nDone.")

def convert_to_tensor():
    print("\nConvert MRI from *.nii to PyTorch tensor")
    brainmask_path = glob(FREESURFER_PATH + '/**/brainmask.nii', recursive=True)

    for idx in tqdm(range(len(brainmask_path))):
        nii_path = brainmask_path[idx]
        # new filename
        filename1 = nii_path[nii_path.rindex("ADNI_")+5:nii_path.rindex("ADNI_")+15]
        filename2 = nii_path[nii_path.rindex("_I")+1:nii_path.rindex("/mri/")]
        filename = filename1 + "_" + filename2

        # savepath
        savepath = os.path.join(SAVE_PATH, filename)

        # read MRI image
        img = np.array(nib.load(nii_path).get_fdata())
        img = torch.tensor(img, dtype=torch.float32)
        img = img[None,:,:,:]

        # save MRI image
        torch.save(img, savepath)
    print("Convert Done.")

def main():

    # step 1: copy ADNI MRI data to a FreeSurfer working folder
    copy_adni_to_freesurfer()

    # step 2: in the FreeSurfer working folder, skull stripping, save as *.mgz file
    skull_remove()

    # step 3: convert the *.mgz file to *.nii
    convert_mri()

    # step 4: convert the *.nii file to PyTorch tensor
    convert_to_tensor()

if __name__ == "__main__":
    main()
