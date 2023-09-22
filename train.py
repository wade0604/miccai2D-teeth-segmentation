import os, cv2
import numpy as np
import pandas as pd
import random, tqdm

from collections import Counter
import matplotlib.pyplot as plt
import segmentation_models_pytorch.utils as smputils
import segmentation_models_pytorch as smp
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album
warnings.filterwarnings("ignore")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



SEED = 0
seed_everything(SEED)


DATA_DIR = ''
fold_number =5.0
metadata_df = pd.read_csv(os.path.join(DATA_DIR, 'metadata_fusai5fold.csv'))
metadata_pseudo_df = pd.read_csv(os.path.join(DATA_DIR, 'metadata_fusai_unlabel_top1000.csv'))

train_df = metadata_df[metadata_df['split']!=fold_number]
train_df = pd.concat([train_df,metadata_pseudo_df])
train_df = train_df[['image_id', 'image_path', 'mask_path']]
train_df['image_path'] = train_df['image_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
train_df['mask_path'] = train_df['mask_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
train_df['image_path'] = train_df['image_path'].str.replace('\\', '/')
train_df['mask_path'] = train_df['mask_path'].str.replace('\\', '/')

valid_df = metadata_df[metadata_df['split']==fold_number]#五折1.0-5.0
valid_df = valid_df[['image_id', 'image_path', 'mask_path']]
valid_df['image_path'] = valid_df['image_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
valid_df['mask_path'] = valid_df['mask_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
valid_df['image_path'] = valid_df['image_path'].str.replace('\\', '/')
valid_df['mask_path'] = valid_df['mask_path'].str.replace('\\', '/')

print(train_df)
print(valid_df)

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


# helper function for data visualization
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


# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    x = np.argmax(image, axis=-1)
    return x


# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x
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
            image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
            print(Counter(mask.flatten().tolist()))

            # one-hot-encode the mask
            mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
            # print(mask.shape)#(320, 640, 2)

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



def get_training_augmentation():
    transform = album.Compose([
        album.HorizontalFlip(),
        album.VerticalFlip(),
        album.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
                         border_mode=cv2.BORDER_REFLECT),

        album.OneOf([
            album.ElasticTransform(p=.3),
            album.MedianBlur(p=0.3),
            album.MotionBlur(p=0.3),
            album.GaussianBlur(p=.3),
            album.GaussNoise(p=.3),
            album.OpticalDistortion(p=0.3),
            album.GridDistortion(p=.1),
        ], p=0.3),
        album.OneOf([
            album.ColorJitter(p=0.5),
            album.HueSaturationValue(15,25,0),
            album.CLAHE(clip_limit=2),
            album.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3,p=0.75),
        ], p=0.3),
    ],p=0.9)
    return transform



def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)

augmented_dataset = ToothDataset(
    train_df,
    augmentation=get_training_augmentation(),
    class_rgb_values=class_rgb_values,
)
from segmentation_models_pytorch.utils import base
class CustomLoss(base.Loss):

    def __init__(self):
        super(CustomLoss, self).__init__()

        self.diceloss = smp.losses.DiceLoss(mode='binary')
        self.binloss = smp.losses.SoftBCEWithLogitsLoss(reduction='mean', smooth_factor=0.1)

    def forward(self, output, mask):
        dice = self.diceloss(output, mask)
        bce = self.binloss(output, mask)

        loss = dice * 0.7 + bce * 0.3

        return loss

ENCODER = 'mit_b2'
ENCODER_WEIGHTS = 'imagenet'
CLASSES =  ['background', 'tooth']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation


model= smp.MAnet(
    encoder_name=ENCODER,
    encoder_depth=4,
    decoder_channels= (512,256, 128, 64),
    encoder_weights=ENCODER_WEIGHTS,
    # decoder_pab_channels=128,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

# model = torch.load('save model/初赛预训练模型.pth')


preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# Get train and val dataset instances
train_dataset = ToothDataset(
    train_df,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=class_rgb_values,
)

valid_dataset = ToothDataset(
    valid_df,
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=class_rgb_values,
)

# Get train and val data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=2)

TRAINING = True

# Set num of epochs
EPOCHS = 200


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss = CustomLoss()
metrics = [
    smputils.metrics.IoU(threshold=0.5),
]


optimizer = torch.optim.AdamW([
    dict(params=model.parameters(), lr=6e-5,weight_decay=0.01),
])


schedulers = [
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=1, T_mult=2, eta_min=1e-6,),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)]
lr_scheduler =  torch.optim.lr_scheduler.SequentialLR(optimizer,schedulers,milestones=[62])


train_epoch = smputils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smputils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

if TRAINING:

    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    for i in range(0, EPOCHS):

        # Perform training & validation
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)
        lr_scheduler.step()
        print('lr:{}'.format(optimizer.param_groups[0]['lr']))

        if best_iou_score < valid_logs['iou_score']:
            if best_iou_score != 0:
                os.remove(f'save model/best_{ENCODER}_fold_{int(fold_number)}_iou_{best_iou_score:.4f}.pth')
            best_iou_score = valid_logs['iou_score']
            torch.save(model, f'save model/best_{ENCODER}_fold_{int(fold_number)}_iou_{valid_logs["iou_score"]:.4f}.pth')
            print(f'Best Score:{best_iou_score} Model saved!')

