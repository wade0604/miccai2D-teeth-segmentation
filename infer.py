import torch
print(torch.__version__)
print(torch.cuda.is_available())
import os
import cv2
print(cv2.__version__)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from torch.utils.data import DataLoader
import albumentations as album
warnings.filterwarnings("ignore")
print(os.path.abspath('infers_fusai'))
def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(15, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

DATA_DIR = ''




class_dict = pd.read_csv('class_dict.csv')
class_names = class_dict['name'].tolist()
class_rgb_values = class_dict[['r','g','b']].values.tolist()

print('All dataset classes and their corresponding RGB values in labels:')
print('Class Names: ', class_names)
print('Class RGB values: ', class_rgb_values)


def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map

class ToothDataset(torch.utils.data.Dataset):


    def __init__(
            self,
            df,
            class_rgb_values=None,
            augmentation=None,
            preprocessing=None,
            train=True
    ):
        self.image_paths = df['image_path'].tolist()
        self.mask_paths = df['mask_path'].tolist()

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.train = train


    def __getitem__(self, i):
        if self.train:
            # read images and masks
            # print(self.image_paths[i])
            image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)

            # one-hot-encode the mask
            mask = one_hot_encode(mask, self.class_rgb_values).astype('float')#类别编码 像素值变0和1

            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            return image, mask

        else:
            image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image)
                image= sample['image']

            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image)
                image = sample['image']

            return image,self.image_paths[i]


    def __len__(self):
        # return length of
        return len(self.image_paths)



def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')



def to_tensor_mask(x, **kwargs):
    return x.transpose(2, 0, 1).astype(np.int64)

def get_preprocessing():
    _transform = []
    _transform.append(album.Normalize())
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor_mask))

    return album.Compose(_transform)


if __name__ =='__main__':
    # model = smp.MAnet(
    #     encoder_name='mit_b1',
    #     encoder_depth=5,
    #     decoder_channels=(256, 128, 64, 32, 16),
    #     encoder_weights=None,
    #     classes=2,
    #     activation='sigmoid',
    # ).cuda()

    sample_preds_folder = 'infers_fusai/infers/'
    if not os.path.exists(sample_preds_folder):
        os.makedirs(sample_preds_folder)

    path ='save model/best_mit_b2_fold_100_iou_0.9741.pth'

    DEVICE = 'cuda'
    best_model = torch.load(path, map_location=DEVICE)

    test_df = pd.read_csv(os.path.join(DATA_DIR, 'metadata_fusai_test.csv'))
    test_df['image_path'] = test_df['image_path'].str.replace('\\', '/')
    test_dataset1 = ToothDataset(
        test_df,
        preprocessing=get_preprocessing(),
        class_rgb_values=class_rgb_values,
        train=False
    )

    from tqdm import tqdm
    save_probability = np.zeros((1000,320,640))
    for idx in tqdm(range(len(test_dataset1))):

        image1,image_paths1 = test_dataset1[idx]
        print(image_paths1)
        best_model.eval()

        with torch.no_grad():#推理不需要梯度 可以降低内存需求

            x_tensor1 = torch.from_numpy(image1).to(DEVICE).unsqueeze(0)#1,3,320,640


            pred_mask1 = best_model(x_tensor1)#1,2,320,640 原图
            pred_mask1 = pred_mask1[:,0,:]#1,320,640 不需要sigmoid了因为已经在模型最后激活过了

            x_tensor2 = torch.flip(x_tensor1, [2])#上下翻转 相当于0,2
            pred_mask2 = best_model(x_tensor2)
            pred_mask2 = torch.flip(pred_mask2, [2])[:,0,:]

            x_tensor3 = torch.flip(x_tensor1, [3])#左右翻转 相当于0,3
            pred_mask3 = best_model(x_tensor3)
            pred_mask3 = torch.flip(pred_mask3, [3])[:,0,:]


        pred_mask = (pred_mask1 + pred_mask2 + pred_mask3

                     ) / 3.0 #0.9565
        # pred_mask = pred_mask1
        save_probability[idx] = pred_mask.cpu().numpy().reshape(320, 640)
        threshold = 0.5
        pred_mask = torch.where(pred_mask >= threshold, torch.tensor(255, dtype=torch.float).to(DEVICE), pred_mask)#白色
        pred_mask = torch.where(pred_mask < threshold, torch.tensor(0, dtype=torch.float).to(DEVICE), pred_mask)#黑色

        out = pred_mask.detach().cpu().numpy().reshape(1, 320, 640)


        from PIL import Image
        img = Image.fromarray(out[0].astype(np.uint8))
        img = img.convert('1')#0 255转0 1

        img.save(os.path.join(sample_preds_folder, f"test_{idx}.png"))



