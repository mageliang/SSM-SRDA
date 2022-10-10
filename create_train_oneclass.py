import os
from PIL import Image as m
from tqdm import tqdm
import random
import cv2
import numpy as np
# from util import util

# mass train dataset paths
mass_images_path = "./Mass/srda_train"
mass_labels_path = "./Mass/srda_train_label"
mass_image_size = 500

# inria train dataset paths
inria_images_path_train = "./inria_building_datasets/inria_train_images"
inria_labels_path_train = "./inria_building_datasets/inria_train_labels"
inria_image_size = 1000

def createSetsA(image_dir, label_dir, image_size, output_path):
    index = 1
    label_paths = os.listdir(label_dir)
    for path_item in tqdm(label_paths):
        # image = m.open(image_dir + "/" + path_item).convert('RGB')
        image = cv2.imread(image_dir + "/" + path_item + "f", cv2.IMREAD_UNCHANGED)
        # label = m.open(label_dir + "/" + path_item.split(".")[0] + ".tif").convert("L")
        label = cv2.imread(label_dir + "/" + path_item, cv2.IMREAD_UNCHANGED)
        X_height, X_width, _ = image.shape # 原图大小？ 1500
        for key in range(80):
            random_width = random.randint(0, X_width - image_size - 1)
            random_height = random.randint(0, X_height - image_size - 1)
            src_roi = image[random_height: random_height + image_size, random_width: random_width + image_size, :]
            label_roi = label[random_height: random_height + image_size, random_width: random_width + image_size, :]

            cv2.imwrite((output_path + "/images/image%d.tif" % index), src_roi)
            cv2.imwrite((output_path + "/labels/label%d.tif" % index), label_roi)  #500×500
            index += 1


def createSetsB(image_dir, label_dir, image_size, output_path):
    index = 1
    image_paths = os.listdir(image_dir)
    for path_item in tqdm(image_paths):
        # image = m.open(image_dir + "/" + path_item).convert('RGB')
        image = cv2.imread(image_dir + "/" + path_item, cv2.IMREAD_UNCHANGED)
        # label = m.open(label_dir + "/" + path_item.split(".")[0] + ".tif").convert("L")
        label = cv2.imread(label_dir + "/" + path_item, cv2.IMREAD_UNCHANGED)
        X_height, X_width, _ = image.shape
        for row in range(5):
            start_row = row * image_size
            end_row = start_row + image_size
            for colom in range(5):
                start_colom = colom * image_size
                end_colom = start_colom + image_size


                src_roi = image[start_colom: end_colom, start_row: end_row, :]
                # label_roi = label[start_colom: end_colom, start_row: end_row, :]
                label_roi = label[start_colom: end_colom, start_row: end_row]

                cv2.imwrite((output_path + "/images/image%d.tif" % index), src_roi)
                cv2.imwrite((output_path + "/labels/label%d.tif" % index), label_roi)   #1000×1000
                index += 1


def change_labelA(label_dir):
    image_paths = os.listdir(label_dir)
    for path_item in tqdm(image_paths):
        #print(path_item)
        # label = m.open(label_dir + "/" + path_item)#.convert("L")  #如果是二值图 不用转灰度
        label = m.open(label_dir + "/" + path_item).convert("L")
    

        
        # change 255 to 1
        # im_point = label.point(lambda x: x // 255) # 如果是0，255的标签

        im_point = label.point(lambda x: x//76) # 这里76是红色

        im_point.save(label_dir + "/" + path_item)

def change_labelB(label_dir):
    image_paths = os.listdir(label_dir)
    for path_item in tqdm(image_paths):
        # print(path_item)
        # label = m.open(label_dir + "/" + path_item)#.convert("L")  #如果是二值图 不用转灰度
        label = m.open(label_dir + "/" + path_item).convert("L")
        
        # change 255 to 1
        im_point = label.point(lambda x: x // 255) # 如果是0，255的标签

        # im_point = label.point(lambda x: x//76) # 这里76是红色

        im_point.save(label_dir + "/" + path_item)

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    # create the paths of created datasets
    mkdirs(['./mass_inria/trainA/images', './mass_inria/trainA/labels',
                 './mass_inria/trainB/images', './mass_inria/trainB/labels'])
    mass_output_path = "./mass_inria/trainA"
    # createSetsA(mass_images_path, mass_labels_path, mass_image_size, mass_output_path)

    inria_output_path_train = "./mass_inria/trainB"
    # createSetsB(inria_images_path_train, inria_labels_path_train, inria_image_size, inria_output_path_train)

    trainA_label_dir = "./mass_inria/trainA/labels"
    change_labelA(trainA_label_dir)
    
    
    # trainB_label_dir = "./mass_inria/trainB/labels"
    # change_labelB(trainB_label_dir)