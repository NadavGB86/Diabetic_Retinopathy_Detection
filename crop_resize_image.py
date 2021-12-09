import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
import random


def _subtract_local_average(im, target_radius_size):
    # cv2 = tfds.core.lazy_imports.cv2
    image_blurred = cv2.GaussianBlur(im, (0, 0), target_radius_size / 27)
    im = cv2.addWeighted(im, 4, image_blurred, -4, 128)
    return im


def get_image_info(im):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    flat = im_gray.flatten()
    # print(int(np.median(flat)), int(np.mean(flat)*0.5))

    # cv2.imwrite('orig_image.jpg', im)
    # cv2.imwrite('gray_image.jpg', im_gray)
    ret, mask = cv2.threshold(im_gray, int(np.mean(flat) * 0.5), 255, 0)

    # cv2.imwrite('mask_image_1.jpg', mask)

    kernel = np.ones((5, 5), np.float32) / 25
    mask = cv2.filter2D(mask, -1, kernel)

    im_height, im_width = mask.shape
    # cv2.imwrite('mask_image_2.jpg', mask)

    bottom_line = mask[int(im_height * 0.6), :]
    bottom_line = np.where(bottom_line > 220)
    if len(bottom_line[0]) == 0:
        return 0, 0, 0, 1000
    bottom_left = (int(im_height * 0.6), bottom_line[0][0])
    bottom_right = (int(im_height * 0.6), bottom_line[0][-1])

    left_line = mask[:, bottom_left[1]]
    left_line = np.where(left_line > 220)
    if len(left_line[0]) == 0:
        return 0, 0, 0, 1000

    right_line = mask[:, bottom_right[1]]
    right_line = np.where(right_line > 220)
    if len(right_line[0]) == 0:
        return 0, 0, 0, 1000

    top_left = (left_line[0][0], bottom_left[1])
    top_right = (right_line[0][0], bottom_right[1])

    # print('TopLeft, TopRight', top_left, top_right)

    y_center = int(np.mean([np.mean([bottom_left[0], top_right[0]]),
                            np.mean([bottom_right[0], top_left[0]])]))
    x_center = int(np.mean([np.mean([bottom_left[1], top_right[1]]),
                            np.mean([bottom_right[1], top_left[1]])]))

    radius = []
    for y, x in [top_left, top_right, bottom_left, bottom_right]:
        radius.append(np.sqrt((x_center - x) ** 2 + (y_center - y) ** 2))

    return x_center, y_center, np.mean(radius), np.std(radius)


def create_new_image(img, x, y, r, level):
    half_size = 200
    scale_factor = 290 / r  # about half_size * sqrt(2)

    im_width = int(img.shape[1] * scale_factor)
    im_height = int(img.shape[0] * scale_factor)
    dim = (im_width, im_height)

    # resize image
    angle_list = [-30, -15, 0, 15, 30]
    resized_list = list()

    resize_im = cv2.resize(img, dim, interpolation=cv2.INTER_LANCZOS4)
    # center of circle is x, y
    new_x = int(x * scale_factor)
    new_y = int(y * scale_factor)
    image_center = (new_x, new_y)

    if level == 0:
        rot_mat = cv2.getRotationMatrix2D(image_center, random.choice(angle_list), 1.0)
        temp_im = cv2.warpAffine(resize_im, rot_mat, resize_im.shape[1::-1], flags=cv2.INTER_LINEAR)
        temp_im = _subtract_local_average(temp_im, half_size * np.sqrt(2))
        resized_list.append(temp_im)
    else:
        random.shuffle(angle_list)
        for angle in angle_list:
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            temp_im = cv2.warpAffine(resize_im, rot_mat, resize_im.shape[1::-1], flags=cv2.INTER_LINEAR)
            temp_im = _subtract_local_average(temp_im, half_size * np.sqrt(2))
            resized_list.append(temp_im)

    # print('NewX, NewY', new_x, new_y)
    if new_x < half_size or new_y < half_size \
            or (new_x + half_size) > resized_list[0].shape[1] \
            or (new_y + half_size) > resized_list[0].shape[0]:
        # print((new_x + half_size), resized_list[0].shape[1])
        # print((new_y + half_size), resized_list[0].shape[0])
        return None

    cropped_list = list()
    for resized in resized_list:
        resized = np.array(resized)
        cropped_list.append(resized[(new_y - half_size):(new_y + half_size), (new_x - half_size):(new_x + half_size)])

    return cropped_list


# image = cv2.imread(f'data/orig_images/34496_right.jpeg')
# X_center, Y_center, mean_radius, std_radius = get_image_info(image)
# print(X_center, Y_center, mean_radius, std_radius)
# train_image = create_new_image(image, X_center, Y_center, mean_radius, 0)
# # # cv2.imwrite(f'data/train_images/6263_right.jpeg', train_image)
# exit(0)

if __name__ == '__main__':
    orig_image_path = "data/orig_images"
    train_image_path = "data/train_images"
    df_label = pd.read_csv('data/trainLabels.csv', index_col='image')

    files_list = os.listdir(f'{orig_image_path}')
    count = 0
    df = pd.DataFrame(columns=['File name', 'X center', 'Y center', 'Mean radius', 'Std radius'])

    for image_name in tqdm(files_list):
        count += 1
        new_image_name = image_name.replace('.jpeg', '')
        new_image_name = f"{new_image_name}_idx0_class{df_label.loc[image_name.replace('.jpeg', '')]['level']}.jpg"
        if os.path.exists(f'{train_image_path}/{new_image_name}'):
            continue

        image = cv2.imread(f'{orig_image_path}/{image_name}')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        gray_flat = gray.flatten()
        image_mean = np.mean(gray_flat)
        image_std = np.std(gray_flat)
        if (height < 1184) or (width < 1792) or \
                (image_mean < 20) or (image_mean > 120) or (image_std < 20):
            continue

        X_center, Y_center, mean_radius, std_radius = get_image_info(image)
        df = df.append({'File name': image_name, 'X center': X_center, 'Y center': Y_center,
                        'Mean radius': mean_radius, 'Std radius': std_radius}, ignore_index=True)

        # print(count, round(mean_radius, 2), round(std_radius, 2))
        if std_radius < 15:
            image_name = image_name.replace('.jpeg', '')
            new_image_list = create_new_image(image, X_center, Y_center, mean_radius,
                                              df_label.loc[image_name]['level'])
            if new_image_list is not None:
                for index, new_image in enumerate(new_image_list):
                    if new_image is not None:
                        cv2.imwrite(
                            f"{train_image_path}/{image_name}_idx{index}_class{df_label.loc[image_name]['level']}.jpg",
                            new_image)
        # else:
        #     #print(print(image_name, count, round(mean_radius, 2), round(std_radius, 2)))

        # if count > 50:
        #     break

    # print(df.sort_values('Std radius').tail(50))
    #
    # print(df.sort_values('Mean radius'))
