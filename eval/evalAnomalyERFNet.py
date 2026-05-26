# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor
from functions import *

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    anomaly_score_logit_list = []
    anomaly_score_softmax_list = []
    anomaly_score_entropy_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights
    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)
    model = ERFNet(NUM_CLASSES)
    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()
    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print("Model and weights LOADED successfully")
    model.eval()
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()
        with torch.no_grad():
            result = model(images)
        result = result.squeeze(0)
        
        anomaly_result_logit = 1.0 - np.max(result.data.cpu().numpy(), axis=0)
        probs_tensor = torch.nn.functional.softmax(result.data.cpu(), dim=0)  
        anomaly_result_softmax = 1.0 - np.max(probs_tensor.numpy(), axis=0)
        anomaly_result_entropy = -torch.sum(probs_tensor * torch.log(probs_tensor), dim=0).data.cpu().numpy()            
        
        pathGT = create_pathGT(path)
        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = create_oodgts(mask, pathGT)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_logit_list.append(anomaly_result_logit)
             anomaly_score_softmax_list.append(anomaly_result_softmax)
             anomaly_score_entropy_list.append(anomaly_result_entropy)
        del result, anomaly_result_logit, anomaly_result_softmax, anomaly_result_entropy, ood_gts, mask
        torch.cuda.empty_cache()

    file.write( "\n")

    [prc_auc_logit, fpr_logit] = eval_score(ood_gts_list, anomaly_score_logit_list)
    [prc_auc_softmax, fpr_softmax] = eval_score(ood_gts_list, anomaly_score_softmax_list)
    [prc_auc_entropy, fpr_entropy] = eval_score(ood_gts_list, anomaly_score_entropy_list)

    print(f'AUPRC logit score: {prc_auc_logit*100.0}')
    print(f'FPR@TPR95 logit: {fpr_logit*100.0}')

    print(f'AUPRC softmax score: {prc_auc_softmax*100.0}')
    print(f'FPR@TPR95 softmax: {fpr_softmax*100.0}')

    print(f'AUPRC entropy score: {prc_auc_entropy*100.0}')
    print(f'FPR@TPR95 entropy: {fpr_entropy*100.0}')

    file.write(('    AUPRC logit score:' + str(prc_auc_logit*100.0) + '   FPR@TPR95 logit:' + str(fpr_logit*100.0) +
                '\n    AUPRC softmax score:' + str(prc_auc_softmax*100.0) + '   FPR@TPR95 softmax:' + str(fpr_softmax*100.0) +
                '\n    AUPRC entropy score:' + str(prc_auc_entropy*100.0) + '   FPR@TPR95 entropy:' + str(fpr_entropy*100.0)))
    file.close()

if __name__ == '__main__':
    main()
