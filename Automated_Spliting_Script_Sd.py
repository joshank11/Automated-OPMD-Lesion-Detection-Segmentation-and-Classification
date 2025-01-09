import cv2 as cv
import numpy as np
import os

image_folder = '/home/interns/Desktop/sandipan/OPMD_refined_dataset/images'
label_folder = '/home/interns/Desktop/sandipan/OPMD_refined_dataset/labels'

labels = os.listdir(label_folder)
images = os.listdir(image_folder)

labels_sorted = sorted(labels)
images_sorted = sorted(images)

print(labels_sorted)
print(images_sorted)
test_label = '00_1ca83cc9-20221210_181957.txt'
test_image = '00_1ca83cc9-20221210_181957.jpg'
j = 1

for label, image in zip(labels_sorted, images_sorted):
    # break # to avoid the code from running again by mistake
    label_path = os.path.join(label_folder, label)
    image_path = os.path.join(image_folder, image)

    image_file = cv.imread(image_path)

    with open(label_path, 'r') as labelt:
        content = labelt.read()

    splitted_content = content.split('\n')
    element = np.array([list(map(float, line.split())) for line in splitted_content], dtype=object)
    labelz = []
    # coordinates = []
    for img_class in range(len(element) - 1):
        labelz.append(element[img_class][0])
        # coordinates.append(element[img_class][1:])

    version = 0
    print(len(labelz))
    for i in range(len(labelz)):
        img_path = f'/home/interns/Desktop/sandipan/OPMD_Segregated_images/Images/version{version}_{image}'
        label_path = f'/home/interns/Desktop/sandipan/OPMD_Segregated_images/Labels/version{version}_{label}'
        coordinate_path = f'/home/interns/Desktop/sandipan/OPMD_Segregated_images/coordinates/version{version}_{label}'
        coordinates = element[i][1:]

        cv.imwrite(img_path, image_file)
        with open(coordinate_path, 'w') as coordi:
            coordi.write(f'{coordinates}')
        with open(label_path, 'w') as labelboom:
            labelboom.write(f'{labelz[i]}')
        version = version + 1

    # break
# /home/interns/Desktop/sandipan/OPMD_Segregated_images