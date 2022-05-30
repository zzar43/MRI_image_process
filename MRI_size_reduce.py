from tkinter.tix import MAIN
import numpy as np
from glob import glob
from tqdm import tqdm
import os

MAIN_PATH = "/Users/lida/Local_Document/ADNI_dataset/UNet_IDP_Stage_1"
DATA_PATH = os.path.join(MAIN_PATH, 'Dataset')
DATA_SAVEPATH = os.path.join(MAIN_PATH, 'Dataset_reduced')
MRI_PATH_LIST = glob(DATA_PATH + '/**/*.npz', recursive=True)

# find the index for each MRI file
def find_idx(img):
    idx = [0 for i in range(8)]

    img_idx = False
    for i in range(img.shape[1]):
        if img_idx == False and img[:,i,:,:].max() > 0:
            idx[2] = i
            img_idx = True
        if img_idx == True and img[:,i,:,:].max() < 1.:
            idx[3] = i-1
            break

    img_idx = False
    for i in range(img.shape[2]):
        if img_idx == False and img[:,:,i,:].max() > 0:
            idx[4] = i
            img_idx = True
        if img_idx == True and img[:,:,i,:].max() < 1.:
            idx[5] = i-1
            break

    img_idx = False
    for i in range(img.shape[3]):
        if img_idx == False and img[:,:,:,i].max() > 0:
            idx[6] = i
            img_idx = True
        if img_idx == True and img[:,:,:,i].max() < 1.:
            idx[7] = i-1
            break

    return idx

def main():
    # check data size
    # MRI_size = [min_0, max_0, min_1, max_1, min_2, max_2, min_3, max_3]
    MRI_size = [0 for i in range(2*4)]
    MRI_size[2] = MRI_size[4] = MRI_size[6] = 256

    print('Start analysis the MRI image file:')
    loop = tqdm(range(len(MRI_PATH_LIST)), ncols=120)
    for idx in loop:
        img = np.load(MRI_PATH_LIST[idx])['img']
        MRI_size_temp = find_idx(img)

        # dim 1
        MRI_size[2] = min(MRI_size_temp[2], MRI_size[2])
        MRI_size[3] = max(MRI_size_temp[3], MRI_size[3])

        # dim2
        MRI_size[4] = min(MRI_size_temp[4], MRI_size[4])
        MRI_size[5] = max(MRI_size_temp[5], MRI_size[5])

        # dim3
        MRI_size[6] = min(MRI_size_temp[6], MRI_size[6])
        MRI_size[7] = max(MRI_size_temp[7], MRI_size[7])

        loop.set_postfix(size=MRI_size)
        
    print('Analysis done. MRI datasize: ', MRI_size)

    # reduce data size
    MRI_size[2] -= 2
    MRI_size[4] -= 2
    MRI_size[6] -= 2
    MRI_size[3] += 2
    MRI_size[5] += 2
    MRI_size[7] += 2
    print('Reduce done. New data size: ', MRI_size)

    # read file and reduce the size
    loop = tqdm(range(len(MRI_PATH_LIST)), ncols=120)
    for idx in loop:
        img = np.load(MRI_PATH_LIST[idx])['img']

        img = img[0, MRI_size[2]:MRI_size[3], MRI_size[4]:MRI_size[5], MRI_size[6]:MRI_size[7]]
        img = img[None,:,:,:]

        filename = os.path.split(MRI_PATH_LIST[idx])[1]
        savename = os.path.join(DATA_SAVEPATH, filename)
        
        np.savez_compressed(savename, img=img)
        loop.set_postfix(filename = filename)

    print('All done.')

if __name__ == "__main__":
    main()
