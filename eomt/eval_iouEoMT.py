# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from models.eomt import EoMT 
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry
from evalAnomalyEomt import *

NUM_CHANNELS = 3
NUM_CLASSES = 20

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize((1024,1024), Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize((1024,1024), Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   #ignore label to 19
])

def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_eomt(args, device)

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")


    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)


    iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            result = model(images)
    
        mask_logits_last_layer = result[0][-1] # Just the last layer's mask of the transformer
        class_logits_last_layer = result[1][-1]

        # Expands the output mask dimensions of the model to match the ground truth's
        mask_logits_last_layer = torch.nn.functional.interpolate(
            mask_logits_last_layer,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
        )

        mask_prob = torch.sigmoid(mask_logits_last_layer) 
        class_prob = torch.softmax(class_logits_last_layer, dim=-1)

        B, Q, C = class_prob.shape
        _, _, H, W = mask_prob.shape
        cp = class_prob.transpose(1, 2) # (B, C, Q)
        mp = torch.flatten(input = mask_prob,start_dim = 2) # (B, Q, H*W)
        # This operation is performed for each batch
        pixel_logits = torch.matmul(cp, mp) # (B, C, H*W)
        pixel_logits = pixel_logits.unflatten(2, (H, W)) # (B, C, H, W)
        # Per ogni pixel, score per classe
        pixel_logits = pixel_logits.squeeze(0) # (C, H, W)


        iouEvalVal.addBatch(pixel_logits.max(0)[1].unsqueeze(0).unsqueeze(0).data, labels)

        filenameSave = filename[0].split("leftImg8bit/")[1] 

        print (step, filenameSave)


    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    #print("TOTAL IOU: ", iou * 100, "%")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Road")
    print(iou_classes_str[1], "sidewalk")
    print(iou_classes_str[2], "building")
    print(iou_classes_str[3], "wall")
    print(iou_classes_str[4], "fence")
    print(iou_classes_str[5], "pole")
    print(iou_classes_str[6], "traffic light")
    print(iou_classes_str[7], "traffic sign")
    print(iou_classes_str[8], "vegetation")
    print(iou_classes_str[9], "terrain")
    print(iou_classes_str[10], "sky")
    print(iou_classes_str[11], "person")
    print(iou_classes_str[12], "rider")
    print(iou_classes_str[13], "car")
    print(iou_classes_str[14], "truck")
    print(iou_classes_str[15], "bus")
    print(iou_classes_str[16], "train")
    print(iou_classes_str[17], "motorcycle")
    print(iou_classes_str[18], "bicycle")
    print("=======================================")
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())
