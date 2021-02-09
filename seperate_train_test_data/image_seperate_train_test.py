import glob
import os
from PIL import Image
import random
import shutil
import numpy as np

save_path = "./seperate/"
save_path_train = "./seperate/train/"             # set the directory path
save_path_test = "./seperate/test/"
if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(save_path_train):
    os.mkdir(save_path_train)
if not os.path.exists(save_path_test):
    os.mkdir(save_path_test)

                                         # set original image path ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
data_path = sorted(glob.glob(os.environ['HOME'] + "/Dataset/car/data/image/*/*/*/*"))
random.shuffle(data_path)
#label_path = [ label.replace("image", "label").replace(".jpg", ".txt") for label in data_path]

#quit()
count = 0
test_image_number = 1000
for idx, image_file in enumerate(data_path[:10000]):

    #print("!")
    if idx % 5000 == 0:
        print(idx, " images processed")
        #print(counter_front, counter_frontside, counter_side)
    label_file = image_file.replace("image", "label").replace(".jpg", ".txt")
    crop_rate = 0.0
    flag_skip = False
    flag_front_rear = False
    for i, line in enumerate(open(label_file, "r")):
        if (i == 0 and line.split()[0] == "-1"):
            flag_skip = True
            break
        elif (i == 0 and line.split()[0] == "3"):
            crop_rate = 0.15
        elif (i == 0 and (line.split()[0] == "4" or line.split()[0] == "5")):
            crop_rate = 0.1
        elif (i == 0 and (line.split()[0] == "1" or line.split()[0] == "2")):
            flag_front_rear = True

        if i == 2:
            coordinate = line.split()
            coordinate = list(map(int, coordinate))
            x1, y1 = coordinate[0], coordinate[1]
            x2, y2 = coordinate[2], coordinate[3]

    if flag_skip:
        continue

    color_img = Image.open(image_file)
    img = np.asarray(color_img)

    bounging_box_width = (x2 - x1)
    bounding_box_height = (y2 - y1)
    center_boundingbox_x = int((x1 + x2) / 2.0)
    center_boundingbox_y = int((y1 + y2) / 2.0)
    center_image_x = int(color_img.size[0] / 2.0)
    center_image_y = int(color_img.size[1] / 2.0)
    difference_x = abs(center_boundingbox_x - center_image_x)
    difference_y = abs(center_boundingbox_y - center_image_y)


    if (difference_x > 0.3 * bounging_box_width or difference_y > 0.5 * bounding_box_height):
        print("not center!!!    ", idx)
        continue

    if count >= test_image_number:
        shutil.copy(image_file, save_path_train + str(idx) + ".jpg")
    else:
        shutil.copy(image_file, save_path_test + str(idx) + ".jpg")

    count += 1