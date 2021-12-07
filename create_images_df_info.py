import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import cv2
import pandas_profiling

orig_image_path = "data/orig_images"
train_image_path = "data/train_images"
df_label = pd.read_csv('data/trainLabels.csv', index_col='image')

files_list = os.listdir(f'{orig_image_path}')
count = 0
df_label = pd.read_csv('data/trainLabels.csv', index_col='image')
df = pd.DataFrame(columns=['File name', 'Patient Id', 'Side', 'Class', 'Width', 'Height', 'Ratio', 'Image mean', 'Image std'])

for image_name in tqdm(files_list):
    count += 1
    image = cv2.imread(f'{orig_image_path}/{image_name}')
    patient_id, side = image_name.replace('.jpeg', '').split('_')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    ratio = round((width / height), 4)

    gray_flat = gray.flatten()
    image_mean = np.mean(gray_flat)
    image_std = np.std(gray_flat)

    df = df.append({'File name': image_name, 'Patient Id': patient_id, 'Side': side,
                    'Class': df_label.loc[image_name.replace('.jpeg', '')]['level'],
                    'Width': width, 'Height': height, 'Ratio': ratio,
                    'Image mean': image_mean, 'Image std': image_std},
                   ignore_index=True)

    # if count > 100:
    #     break

profile = df.profile_report(title="Images Profiling Report")
profile.to_file("images_profile_data.html")

print(df.sample(50))
