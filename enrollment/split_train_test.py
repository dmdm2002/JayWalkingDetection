import os
import re
import math
import random
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

random.seed(1004)


img_names = os.listdir('C:/Users/rlawj/sample_DB/Sejong/segmentation/images')
cctv_names = []
for name in img_names:
    cctv = name.split('_')[0]
    cctv_names.append(cctv)

cctv_names = list(set(cctv_names))

random.shuffle(cctv_names)
train_len = math.ceil(len(cctv_names) * 0.80)
train_cctv_names = cctv_names[:train_len]
test_cctv_names = cctv_names[train_len:]

print(len(train_cctv_names))
print(len(test_cctv_names))

df_list = []
for name in img_names:
    cctv = name.split('_')[0]
    if cctv in train_cctv_names:
        mask_name = re.compile('.jpg').sub('_mask.png', name)
        df_list.append(['train', name, mask_name])
    else:
        mask_name = re.compile('.jpg').sub('_mask.png', name)
        df_list.append(['test', name, mask_name])

df = pd.DataFrame(df_list, columns=['run_type', 'img_name', 'mask_name'])

df.to_csv('C:/Users/rlawj/sample_DB/Sejong/segmentation/train_test_split.csv', index=False)
print(df)
print(df['run_type'].value_counts())

# 이렇게 하기 보다는 augmentation한 image를 train으로 사용하고 원본 이미지를 test로 사용하는 것이 좀 더 나을듯?
# data가 너무 적어...

# 아니다 걍 80% train / 20% test로 사용하자
