import numpy as np
import  os
import  pandas as pd
import glob
from sklearn.model_selection import StratifiedKFold


tta_pred_mask1 = np.load('save probability/xxx.npy')
print(tta_pred_mask1.shape)
uncertainties = {}
for i in range(2000):
    read_data = tta_pred_mask1[i]
    uncertainty = -np.sum(read_data*np.log(read_data))/np.sum(read_data)
    uncertainties[i] = uncertainty


df = pd.DataFrame(columns=['image_id', 'fold', 'image_path', 'mask_path'],index=range(1000))

for i in range(1000):#前1000个置信度高的
    print(sorted(uncertainties.items(),key=lambda x:x[1],reverse=False)[i])#越高越不好
    df['image_id'].iloc[i]= sorted(uncertainties.items(),key=lambda x:x[1],reverse=False)[i][0]
    df['image_path'].iloc[i] = f'fusai_train/train/unlabelled/image/train_ul_{sorted(uncertainties.items(),key=lambda x:x[1],reverse=False)[i][0]}.png'
    df['mask_path'].iloc[i] = f'infers_fusai_pseudo/infers/train_ul_{sorted(uncertainties.items(), key=lambda x: x[1], reverse=False)[i][0]}.png'





# 按照image_id列升序排序
df = df.sort_values(by='image_id')
# 重置索引
df = df.reset_index(drop=True)
print(df)


# 这里使用StratifiedKFold来保持分层性质
num_folds = 5
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# 创建一个列表来保存每个样本所在的折叠
image_files = df['image_path']

fold_indices = []
for train_index, test_index in skf.split(image_files, [0] * len(image_files)):
    fold_indices.append(test_index)

# 将折叠索引添加到数据框中
for fold_num, test_indices in enumerate(fold_indices):
    df.loc[test_indices, 'split'] = fold_num + 1

print(df['split'].value_counts())

# 保存为CSV文件
df.to_csv('metadata_fusai_unlabel_top1000.csv', index=False)



