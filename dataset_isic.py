import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self).__init__()
        self.dataset_name = config.datasets
        
        if train:
            images_list = sorted(os.listdir(path_Data+'train/images/'))
            masks_list = sorted(os.listdir(path_Data+'train/masks/'))
            self.data = []
            
            # 处理ISIC2018数据集
            if self.dataset_name == 'ISIC2018':
                for img_file in images_list:
                    if img_file.endswith('.png'):
                        img_id = img_file.split('.')[0]  # 获取图像ID（不带扩展名）
                        mask_file = f"{img_id}.png"  # ISIC2018掩码命名格式
                        if mask_file in masks_list:
                            img_path = path_Data+'train/images/' + img_file
                            mask_path = path_Data+'train/masks/' + mask_file
                            self.data.append([img_path, mask_path])
            # 处理ISIC2017数据集
            else:
                for img_file in images_list:
                    if img_file.endswith('.jpg'):
                        img_id = img_file.split('.')[0]
                        mask_file = f"{img_id}_segmentation.png"  # ISIC2017掩码命名格式
                        if mask_file in masks_list:
                            img_path = path_Data+'train/images/' + img_file
                            mask_path = path_Data+'train/masks/' + mask_file
                            self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer
        else:
            images_list = sorted(os.listdir(path_Data+'val/images/'))
            masks_list = sorted(os.listdir(path_Data+'val/masks/'))
            self.data = []
            
            # 处理ISIC2018数据集
            if self.dataset_name == 'ISIC2018':
                for img_file in images_list:
                    if img_file.endswith('.png'):
                        img_id = img_file.split('.')[0]  # 获取图像ID（不带扩展名）
                        mask_file = f"{img_id}.png"  # ISIC2018掩码命名格式
                        if mask_file in masks_list:
                            img_path = path_Data+'val/images/' + img_file
                            mask_path = path_Data+'val/masks/' + mask_file
                            self.data.append([img_path, mask_path])
            # 处理ISIC2017数据集
            else:
                for img_file in images_list:
                    if img_file.endswith('.jpg'):
                        img_id = img_file.split('.')[0]
                        mask_file = f"{img_id}_segmentation.png"  # ISIC2017掩码命名格式
                        if mask_file in masks_list:
                            img_path = path_Data+'val/images/' + img_file
                            mask_path = path_Data+'val/masks/' + mask_file
                            self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk, img_path

    def __len__(self):
        return len(self.data) 