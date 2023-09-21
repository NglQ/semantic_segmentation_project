import numpy as np
from scipy import ndimage
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import os
from data_loader import load_data_from_names


def overlapping_area(l1, r1, l2, r2):
    x = 0
    y = 1
    ''' Length of intersecting part i.e  
        start from max(l1[x], l2[x]) of  
        x-coordinate and end at min(r1[x], 
        r2[x]) x-coordinate by subtracting  
        start from end we get required  
        lengths '''
    x_dist = abs(min(r1[x], r2[x]) - max(l1[x], l2[x]))

    y_dist = abs(min(r1[y], r2[y]) - max(l1[y], l2[y]))

    area_i = 0
    if x_dist > 0 and y_dist > 0:
        area_i = x_dist * y_dist

    return area_i


def create_training_set(target_max=True):

    rows, cols = 128, 128
    win_rows, win_cols = 5, 5
    threshold = .5 * (2 * rows * 2 * cols) if target_max else .2 * (2 * rows * 2 * cols)

    target_path_annotation_train = './ign_new_min/annotations/training/'
    target_path_images_train = './ign_new_min/images/training/'

    # TODO Start over with new setup using the minimum of the variance
    image_names = os.listdir('./ign/images/training')[500:600]
    mask_names = os.listdir('./ign/annotations/training')[500:600]

    image_names.sort()
    mask_names.sort()

    cnt = 0

    x = load_data_from_names('./ign/images/training', image_names, shape=(256, 256), resize_image=False)
    y = load_data_from_names('./ign/annotations/training', mask_names, shape=(256, 256), mask=True, resize_image=False)
    for ((img, img_name), (mask, mask_name)) in zip(zip(x, image_names), zip(y, mask_names)):
        win_mean = ndimage.uniform_filter(np.argmax(mask, axis=-1), (win_rows, win_cols))
        win_sqr_mean = ndimage.uniform_filter(np.argmax(mask, axis=-1)**2, (win_rows, win_cols))
        win_var = win_sqr_mean - win_mean**2

        mx_place = np.where(win_var == win_var.max()) if target_max else np.where(win_var == win_var.min())

        old_patches = []

        zip_var = zip(mx_place[0], mx_place[1]) if target_max else zip(mx_place[0][:250], mx_place[1][:250])

        for k, (i, j) in enumerate(zip_var):
            x_min = i-rows
            x_max = i+rows
            y_min = j-cols
            y_max = j+cols
            if x_min < 0:
                x_min = 0
                x_max = 2*rows
            if x_max > 1000:
                x_max = 1000
                x_min = 1000 - 2*rows
            if y_min < 0:
                y_min = 0
                y_max = 2*cols
            if y_max > 1000:
                y_max = 1000
                y_min = 1000 - 2*cols

            win_var_1 = np.argmax(mask, axis=-1)[x_min:x_max, y_min:y_max]
            img_1_var = img[x_min:x_max, y_min:y_max]*255

            if old_patches == []:
                cv2.imwrite(f'{target_path_images_train}{img_name[:img_name.find(".")]}_{k}.png', img_1_var)
                cv2.imwrite(f'{target_path_annotation_train}{mask_name[:mask_name.find(".")]}_{k}.png', win_var_1)
                old_patches.append((k, (x_min, x_max), (y_min, y_max)))
            else:
                a = [overlapping_area((x_min, y_min), (x_max, y_max), (old_x_min, old_y_min), (old_x_max, old_y_max)) < threshold
                     for (kk, (old_x_min, old_x_max), (old_y_min, old_y_max)) in old_patches]
                if all(a):
                    cv2.imwrite(f'{target_path_images_train}{img_name[:img_name.find(".")]}_{k}.png', img_1_var)
                    cv2.imwrite(f'{target_path_annotation_train}{mask_name[:mask_name.find(".")]}_{k}.png', win_var_1)
                    old_patches.append((k, (x_min, x_max), (y_min, y_max)))

        if cnt % 10 == 9:
            print(cnt, 'out of', len(image_names))
        cnt += 1


def create_test_set():

    target_path_annotation_val = './ign_new_min/annotations/validation/'
    target_path_images_val = './ign_new_min/images/validation/'

    image_names = os.listdir('./ign/images/validation')
    mask_names = os.listdir('./ign/annotations/validation')

    image_names.sort()
    mask_names.sort()

    x = load_data_from_names('./ign/images/validation', image_names, shape=(256, 256))
    y = load_data_from_names('./ign/annotations/validation', mask_names, shape=(256, 256), mask=True)

    for ((img, img_name), (mask, mask_name)) in zip(zip(x, image_names), zip(y, mask_names)):
        cv2.imwrite(f'{target_path_images_val}{img_name}', img*255)
        cv2.imwrite(f'{target_path_annotation_val}{mask_name}', np.argmax(mask, axis=-1))


def create_dataset():
    create_training_set(target_max=False)
    # create_test_set()


def filter_dataset(data, masks):

    filtered_data = []
    filtered_masks = []

    flattened_masks = np.argmax(masks, axis=-1)
    if 5 in flattened_masks or 6 in flattened_masks:
        filtered_data.append(data)
        filtered_masks.append(masks)

    return filtered_data, filtered_masks


def merge_datasets():
    target_path_annotation_train = './ign_merged/annotations/training/'
    target_path_images_train = './ign_merged/images/training/'

    image_names = os.listdir('./ign_new/images/training')
    mask_names = os.listdir('./ign_new/annotations/training')

    image_names.sort()
    mask_names.sort()

    x = load_data_from_names('./ign_new/images/training', image_names, shape=(256, 256))
    y = load_data_from_names('./ign_new/annotations/training', mask_names, shape=(256, 256), mask=True)

    cnt = 0
    for ((img, img_name), (mask, mask_name)) in zip(zip(x, image_names), zip(y, mask_names)):
        cv2.imwrite(f'{target_path_images_train}{img_name}', img*255)
        cv2.imwrite(f'{target_path_annotation_train}{mask_name}', np.argmax(mask, axis=-1))

        if cnt % 10 == 9:
            print(cnt, 'out of', len(image_names))
        cnt += 1
