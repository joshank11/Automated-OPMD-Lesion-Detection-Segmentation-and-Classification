import numpy as np
import cv2 as cv
# import matplotlib.pyplot as plt
import os

image_folder = '/home/interns/Desktop/sandipan/OPMD_Segregated_images/Images'
images = sorted(os.listdir(image_folder))

label_folder = '/home/interns/Desktop/sandipan/OPMD_Segregated_images/Labels'
labels = sorted(os.listdir(label_folder))

coordinates_folder = '/home/interns/Desktop/sandipan/OPMD_Segregated_images/coordinates'
coordinates = sorted(os.listdir(coordinates_folder))

# print(images)
# print(labels)
# print(coordinates)

target_image_folder = '/home/interns/Desktop/sandipan/OPMD_new_set/Images'
target_label_folder = '/home/interns/Desktop/sandipan/OPMD_new_set/Labels'
target_ground_truth_folder = '/home/interns/Desktop/sandipan/OPMD_new_set/Mask'


def resize_coordinate(img_coordinates, x_shape, y_shape):
    return (img_coordinates * np.array([y_shape, x_shape])).astype(int)

for image_file, label_file, coordinate_file in zip(images, labels, coordinates):
    image_path = os.path.join(image_folder, image_file)
    label_path = os.path.join(label_folder, label_file)
    coordinates_path = os.path.join(coordinates_folder, coordinate_file)

    image = cv.imread(image_path)

    with open(label_path, 'r') as lbl:
        label = lbl.read()

    with open(coordinates_path, 'r') as coor:
        coori = coor.read()
        coorz = coori.strip("[]")
        coord = coorz.split(",")

    coordinates = np.array([float(values) for values in coord]).reshape(-1, 2)

    weight_img = np.zeros_like(image)

    coordinates_resized = resize_coordinate(coordinates, image.shape[0], image.shape[1])

    cv.fillPoly(weight_img, [coordinates_resized], (255, 255, 255))

    # ano = cv.addWeighted(image, 1, weight_img, 0.5, 0)
    # x, y, w, h = cv.boundingRect(coordinates_resized)

    # print('X: ', x, " Y: ", y, " W: ", w, " H: ", h)
    # print('ImageX :', image.shape[1], 'ImageY :', image.shape[0])

    # if w < 40 or h < 40:  # very short bounding boxes, must be an issue with coordinates provided already
    #     continue

    # cropped_image = image[y - 20:y + h + 20, x - 20: x + h + 20]
    # ground_truth_mask = weight_img[y - 20:y + h + 20, x - 20: x + h + 20]

    # if cropped_image.shape[0] <= 0 or cropped_image.shape[1] <=0:
        # continue
    # crp_img = cv.resize(cropped_image, (256, 256))
    # gt_img = cv.resize(ground_truth_mask, (256, 256))

    cropped_image_path = os.path.join(target_image_folder, image_file)
    ground_truth_path = os.path.join(target_ground_truth_folder, image_file)
    label_path = os.path.join(target_label_folder, label_file)

    cv.imwrite(cropped_image_path, image)
    cv.imwrite(ground_truth_path, weight_img)

    with open(label_path, 'w') as label_final:
        label_final.write(label)

    # cv.imshow('real', image)
    # cv.imshow('cropped', crp_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # break
