# MICCAI 2023 Challenges ：STS-基于2D 全景图像的牙齿分割任务
# 初赛第一 复赛第四方案分享 [比赛链接](https://tianchi.aliyun.com/competition/entrance/532086/information)
# 前言
主要用的是segmentation-models-pytorch这一个语义分割库，这个库对新手非常友好，内置了许多主流的Backbone和SegHead。
其实目前工业界主流的还是用mmsegmentation，这个库基于mmcv构建，模型更加全面，但是这个库的AIP接口太高级了，改动起来有点麻烦，对于新手不是很友好。
## 初赛模型
   初赛包含2000张有标签数据与500张测试数据
   Backbone部分使用了MixVisionTransformer(mit-b1、mit-b2、mit-b3)，分割头一开始用的是DeeplabV3+，
   但是效果不太好，后面尝试了不同的分割头发现Manet的效果是最好的。

    ENCODER = 'mit_b1'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES =  ['background', 'tooth']
    ACTIVATION = 'sigmoid' 
    model= smp.MAnet(
        encoder_name=ENCODER,
        encoder_depth=5,
        decoder_channels= (256, 128, 64, 32, 16),
        encoder_weights=ENCODER_WEIGHTS,
        # decoder_pab_channels=128,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )
## 数据增强
   数据增强部分使用albumentations库，包括RandomFlip,ShiftScaleRotate,ElasticTransform,GaussianBlury以及一些颜色对比度变化如
   ColorJitter和RandomBrightnessContrast。由于图片尺寸固定为320*640，因此没有使用Crop和Resize操作。  
   尝试了Cutout、Cutmix和classmix操作，但是好像并没有多大提升。
## 损失函数
   损失函数部分使用了Dice和SoftBCEloss的组合，比例系数为7:3
    
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
## 训练策略
   Epoch为200，优化器使用AdamW，初始学习率为1e-4，采用组合学习率策略如下:
    
    schedulers = [
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=1, T_mult=2, eta_min=1e-6,),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)]
    lr_scheduler =  torch.optim.lr_scheduler.SequentialLR(optimizer,schedulers,milestones=[62])
## TTA
   推理时对原始图像分别进行水平镜像和垂直镜像预测，然后再翻转回去，最后取三次预测结果的平均作为最终结果

    x_tensor1 = torch.from_numpy(image1).to(DEVICE).unsqueeze(0)
    pred_mask1 = best_model(x_tensor1)
    pred_mask1 = pred_mask1[:,0,:]

    x_tensor2 = torch.flip(x_tensor1, [2])
    pred_mask2 = best_model(x_tensor2)
    pred_mask2 = torch.flip(pred_mask2, [2])[:,0,:]

    x_tensor3 = torch.flip(x_tensor1, [3])
    pred_mask3 = best_model(x_tensor3)
    pred_mask3 = torch.flip(pred_mask3, [3])[:,0,:]
    
## 模型融合
   保存不同模型对每张图像的概率logits，进行平均加权融合

## 初赛模型线上分数如下

|mit-b1-Manet-depth5| mit-b2-Manet-depth4 | mit-b3-Manet-depth4 | 初赛模型集成 |
|-------------------|---------------------|---------------------|--------|
| 0.9571          | 0.9568              | 0.9558              |0.9579|

## 复赛模型
   复赛与初赛不同，包含900张有标签数据、2000张无标签数据和500张测试数据  
   直接使用初赛集成模型在复赛排行榜上可以达到0.9599，已经是一个不错的分数了。  


复赛首先将初赛的模型作为预训练权重，一开始我并没有使用无标签数据， 只使用900张有标签数据训练的分数如下:

| mit-b1-Manet-depth5 | mit-b2-Manet-depth4 | mit-b3-Manet-depth4 | 集成     |
|---------------------|---------------------|---------------------|--------|
| 0.9605              | 0.9616              | 0.9605              | 0.9618 |
    

   然后我使用mit-b2-Manet-depth4模型对无标签数据进行推理，同时为了减少低质量的伪标签选择了置信度前Top1000张伪标签加入训练集,伪代码如下:

    tta_pred_mask = np.load('save probability/xxx.npy')#使用初赛模型得到的无标签数据的概率logits
    uncertainties = {}
    for i in range(2000):
        read_data = tta_pred_mask1[i]
        uncertainty = -np.sum(read_data*np.log(read_data))/np.sum(read_data)#计算信息熵
        uncertainties[i] = uncertainty

复赛阶段我使用了五折CV，将训练集随机划分五折训练五次。由于提交次数限制和时间关系，我只选择了mit-b2-Manet-depth4这个模型，
最后在加入伪标签数据之后，5个mit-b2-Manet-depth4模型融合的结果在复赛测试集上达到了0.9621(其实提升并不是很明显)。


## 失败的尝试
* 复赛初期我把大部分时间都花在了端到端的半监督模型尝试上，我尝试了使用mean-teacher架构的[U2PL](https://github.com/Haochen-Wang409/U2PL)对比学习损失以及一致性损失，
但是并没有得到一个很好的效果。反而浪费了很多时间，最后发现还不如分两个阶段打伪标签效果来的好。
* 后处理：尝试使用了cv2的膨胀腐蚀操作和去除孔洞操作，反而掉点了，可能是参数设置的有问题。
* 在初赛阶段还使用了transformers库中号称SOTA效果的Segformer，但是对于这个数据集并没有多少效果，在初赛排行榜上只有0.955左右。
* 使用mmseg的Swin-B-FCN模型和Swin-B-Uperhead模型，效果不佳且需要算力要求较高，放弃。

## 可能有效的方法
* 二分类阈值调参，不一定0.5的效果最佳
* 考虑多尺度推理TTA以及滑动窗口推理TTA