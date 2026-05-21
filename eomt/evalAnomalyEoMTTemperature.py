# Copyright (c) OpenMMLab. All rights reserved.
import os
import yaml
import torch
import numpy as np
import warnings
import glob

from lightning import seed_everything
from torch.nn import functional as F
from argparse import ArgumentParser
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from functions import *


IGNORE_INDEX = 255

input_transform = Compose([
    Resize((1024, 1024), Image.BILINEAR), 
    ToTensor()
])

target_transform = Compose([
        Resize((1024, 1024), Image.NEAREST),
    ])


def main():
    parser = ArgumentParser()
    parser.add_argument("--input")  
    args = parser.parse_args()

    seed_everything(0, verbose=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # TODO: change to the GPU you want to use
    config_path = 'configs/dinov2/cityscapes/semantic/eomt_base_640.yaml' 
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    state_dict_path = '/content/drive/MyDrive/Colab Notebooks/eomt_cityscapes.bin'
    
    warnings.filterwarnings("ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )

    model = load_model(device, config, state_dict_path)
    
    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'w')

    anomaly_score_softmax_list = []
    ood_gts_list = []

    t_vec = np.concatenate((np.array((0.5,0.75,1.1)), np.exp(np.linspace(np.log(0.1), np.log(50), 20))))    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        if device == 'cpu':
            image = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float()
        else:
            image = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()
        image = image.squeeze(0)
        image = (image * 255).to(torch.uint8)
        pixel_logits = compute_logits(image, device, model)
        
        anomaly_result_list = []
        for t in t_vec:
            probs_tensor = torch.nn.functional.softmax(pixel_logits.data.cpu() / t, dim=0)  
            anomaly_result_softmax = 1.0 - np.max(probs_tensor.numpy(), axis=0)
            anomaly_result_list.append(anomaly_result_softmax)    

        pathGT = create_pathGT(path)  
        mask = Image.open(pathGT)
        mask = target_transform(mask)
        # ground truth mask
        ood_gts = create_oodgts(mask, pathGT)
        if 1 not in np.unique(ood_gts):
            continue           
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_softmax_list.append(anomaly_result_list)
        del result, anomaly_result_softmax, ood_gts, mask
        if device == 'cuda':
            torch.cuda.empty_cache()

    file.write( "\n")

    scores_array = np.array(anomaly_score_softmax_list)
    auprc_list = []
    fpr_list = []
    for i,t in enumerate(t_vec):
        [prc_auc_softmax, fpr_softmax] = eval_score(ood_gts_list, scores_array[:, i])
        auprc_list.append(prc_auc_softmax)
        fpr_list.append(fpr_softmax)
        if i <= 2:
            file.write(f'\n  t = {t} -->  ' + 'AUPRC softmax score:' + str(prc_auc_softmax*100.0) + '   FPR@TPR95 softmax:' + str(fpr_softmax*100.0))
            print(f't = {t}')
            print(f'AUPRC softmax score: {prc_auc_softmax*100.0}')
            print(f'FPR@TPR95 softmax: {fpr_softmax*100.0}')
    
    performance_array = np.array(auprc_list) - np.array(fpr_list) + 1.0
    best_index = np.argmax(performance_array)
    best_t = t_vec[best_index]
    best_auprc = auprc_list[best_index]
    best_fpr = fpr_list[best_index]
    file.write(f'\n best t = {best_t} -->  ' + 'AUPRC softmax score:' + str(best_auprc*100.0) + '   FPR@TPR95 softmax:' + str(best_fpr*100.0))
    
    file.close()

if __name__ == '__main__':
    main()
