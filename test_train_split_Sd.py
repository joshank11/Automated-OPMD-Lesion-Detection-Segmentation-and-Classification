import os
import cv2 as cv

image_folder = '/home/interns/Desktop/sandipan/OPMD_new_set/Images'
label_folder = '/home/interns/Desktop/sandipan/OPMD_new_set/Labels'
mask_folder = '/home/interns/Desktop/sandipan/OPMD_new_set/Mask'

images = sorted(os.listdir(image_folder))
labels = sorted(os.listdir(label_folder))
masks = sorted(os.listdir(mask_folder))

# print(type(images[0][9:11])
target_train_folder = {'image_path': '/home/interns/Desktop/sandipan/OPMD_FULL_Split/Train/Images',
                      'label_path': '/home/interns/Desktop/sandipan/OPMD_FULL_Split/Train/Labels',
                      'mask_path': '/home/interns/Desktop/sandipan/OPMD_FULL_Split/Train/Masks'}

target_test_folder = {'image_path': '/home/interns/Desktop/sandipan/OPMD_FULL_Split/Test/Images',
                       'label_path': '/home/interns/Desktop/sandipan/OPMD_FULL_Split/Test/Labels',
                       'mask_path': '/home/interns/Desktop/sandipan/OPMD_FULL_Split/Test/Masks'}

type_img = ""
for image, mask, label in zip(images, masks, labels):

    if image[9:11] == "00":
        type_img = "test"
    else:
        type_img = "train"

    if type_img == 'train':
        final_image_file = os.path.join(target_train_folder['image_path'], image)
        final_label_file = os.path.join(target_train_folder['label_path'], label)
        final_mask_file = os.path.join(target_train_folder['mask_path'], mask)

        image_file = os.path.join(image_folder, image)
        label_file = os.path.join(label_folder, label)
        mask_file = os.path.join(mask_folder, mask)

        imagex = cv.imread(image_file)
        maskx = cv.imread(mask_file)
        with open(label_file, 'r') as img_class:
            content = img_class.read()

        cv.imwrite(final_image_file, imagex)
        cv.imwrite(final_mask_file, maskx)

        with open(final_label_file, 'w') as label_z:
            label_z.write(content)

    if type_img == 'test':
        final_image_file = os.path.join(target_test_folder['image_path'], image)
        final_label_file = os.path.join(target_test_folder['label_path'], label)
        final_mask_file = os.path.join(target_test_folder['mask_path'], mask)

        image_file = os.path.join(image_folder, image)
        label_file = os.path.join(label_folder, label)
        mask_file = os.path.join(mask_folder, mask)

        imagex = cv.imread(image_file)
        maskx = cv.imread(mask_file)
        with open(label_file, 'r') as img_class:
            content = img_class.read()

        cv.imwrite(final_image_file, imagex)
        cv.imwrite(final_mask_file, maskx)

        with open(final_label_file, 'w') as label_z:
            label_z.write(content)

    # break
