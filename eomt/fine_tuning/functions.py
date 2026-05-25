import numpy as np
import random

import os
import sys
import time

from PIL import Image
from src.dataset.coco import COCO
from pycocotools.coco import COCO as coco_tools

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].

    Adapted from https://github.com/tianyu0207/PEBAL/blob/main/code/dataset/data_loader.py
    """

    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0

        boxes[i] = np.array([y1, x1, y2, x2])

    return boxes.astype(np.int32)


def mix_object(current_labeled_image, current_labeled_mask, cut_object_image, cut_object_mask, ood_label):
    """
    Adapted from Adapted from https://github.com/tianyu0207/PEBAL/blob/main/code/dataset/data_loader.py
    """
    mask = (cut_object_mask == ood_label)
    
    ood_mask = np.expand_dims(mask, axis=2)
    ood_boxes = extract_bboxes(ood_mask)
    ood_boxes = ood_boxes[0, :]  # (y1, x1, y2, x2)
    y1, x1, y2, x2 = ood_boxes[0], ood_boxes[1], ood_boxes[2], ood_boxes[3]
    cut_object_mask = cut_object_mask[y1:y2, x1:x2]
    cut_object_image = cut_object_image[y1:y2, x1:x2, :]

    mask = cut_object_mask == ood_label

    idx = np.transpose(np.repeat(np.expand_dims(cut_object_mask, axis=0), 3, axis=0), (1, 2, 0))

    # if current_labeled_mask.shape[0] != 1024 or current_labeled_mask.shape[1] != 2048:
    #     print('wrong size')
    #     print(current_labeled_mask.shape)
    #     return current_labeled_image, current_labeled_mask

    if mask.shape[0] != 0:
        if current_labeled_mask.shape[0] - cut_object_mask.shape[0] < 0 or \
                current_labeled_mask.shape[1] - cut_object_mask.shape[1] < 0:
            # print('wrong size')
            # print(current_labeled_mask.shape)
            return current_labeled_image, current_labeled_mask
        h_start_point = random.randint(0, current_labeled_mask.shape[0] - cut_object_mask.shape[0])
        h_end_point = h_start_point + cut_object_mask.shape[0]
        w_start_point = random.randint(0, current_labeled_mask.shape[1] - cut_object_mask.shape[1])
        w_end_point = w_start_point + cut_object_mask.shape[1]
    else:
        # print('no odd pixel to mix')
        h_start_point = 0
        h_end_point = 0
        w_start_point = 0
        w_end_point = 0
    
    result_image = current_labeled_image.copy()
    result_image[h_start_point:h_end_point, w_start_point:w_end_point, :][np.where(idx == ood_label)] = \
        cut_object_image[np.where(idx == ood_label)]
    result_label = current_labeled_mask.copy()
    result_label[h_start_point:h_end_point, w_start_point:w_end_point][np.where(cut_object_mask == ood_label)] = \
        cut_object_mask[np.where(cut_object_mask == ood_label)]

    return result_image, result_label


def cut_images():
    start = time.time()
    root = cs_coco_roots.coco_root
    split = "train"
    year = 2017
    id_in = COCO.train_id_in
    id_out = COCO.train_id_out
    min_size = COCO.min_image_size
    annotation_file = '{}/annotations/instances_{}.json'.format(root, split+str(year))
    images_dir = '{}/{}'.format(root, split+str(year))
    tools = coco_tools(annotation_file)
    save_dir = '{}/annotations/ood_seg_{}'.format(root, split+str(year))
    print("\nPrepare COCO{} {} split for OoD training".format(str(year), split))

    # Names of classes that are excluded - these are Cityscapes classes also available in COCO
    exclude_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign']

    # Fetch all image ids that does not include instance from classes defined in "exclude_classes"
    exclude_cat_Ids = tools.getCatIds(catNms=exclude_classes)
    exclude_img_Ids = []
    for cat_Id in exclude_cat_Ids:
        exclude_img_Ids += tools.getImgIds(catIds=cat_Id)
    exclude_img_Ids = set(exclude_img_Ids)
    img_Ids = [int(image[:-4]) for image in os.listdir(images_dir) if int(image[:-4]) not in exclude_img_Ids]

    num_masks = 0
    # Process each image
    print("Ground truth segmentation mask will be saved in:", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("Created save directory:", save_dir)
    for i, img_Id in enumerate(img_Ids):
        img = tools.loadImgs(img_Id)[0]
        h, w = img['height'], img['width']

        # Select only images with height and width of at least min_size
        if h >= min_size and w >= min_size:
            ann_Ids = tools.getAnnIds(imgIds=img['id'], iscrowd=None)
            annotations = tools.loadAnns(ann_Ids)

            # Generate binary segmentation mask
            mask = np.ones((h, w), dtype="uint8") * id_in
            for j in range(len(annotations)):
                mask = np.maximum(tools.annToMask(annotations[j])*id_out, mask)

            # Save segmentation mask
            Image.fromarray(mask).save(os.path.join(save_dir, "{:012d}.png".format(img_Id)))
            num_masks += 1
        print("\rImages Processed: {}/{}".format(i + 1, len(img_Ids)), end=' ')
        sys.stdout.flush()

    # Print summary
    print("\nNumber of created segmentation masks with height and width of at least %d pixels:" % min_size, num_masks)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("FINISHED {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
