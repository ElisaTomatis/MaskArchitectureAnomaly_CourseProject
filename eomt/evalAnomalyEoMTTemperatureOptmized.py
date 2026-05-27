# Copyright (c) OpenMMLab. All rights reserved.
import os
import yaml
import torch
import numpy as np
import warnings
import glob
import shutil

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
    state_dict_path = '/content/drive/MyDrive/ml_anomaly_segmentation/eomt_cityscapes.bin'
    
    warnings.filterwarnings("ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )

    model = load_model(device, config, state_dict_path)
    
    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'w')

    ood_gts_list = []
    valid_paths = []
    # qua faccio lo store dei logits
    os.makedirs('temp_logits', exist_ok=True)
    for path in sorted(glob.glob(os.path.expanduser(str(args.input)))):
        if device == 'cpu':
            image = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float()
        else:
            image = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()
        image = image.squeeze(0)
        image = (image * 255).to(torch.uint8)
        
        # Salva la GT (occupa poco spazio, puoi tenerla in RAM o salvarla)
        pathGT = create_pathGT(path)  
        mask = Image.open(pathGT)
        mask = target_transform(mask)
        # ground truth mask
        ood_gts = create_oodgts(mask, pathGT).astype(np.uint8)
        if 1 not in np.unique(ood_gts):
            print(f"Saltato {path}: nessuna anomalia trovata.")
            continue 
        else:
            pixel_logits = compute_logits(image, device, model).cpu().numpy()
            # SALVIAMO SOLO I VALIDI
            filename = os.path.basename(path)
            np.save(f'temp_logits/{filename}.npy', pixel_logits)
            
            ood_gts_list.append(ood_gts)
            valid_paths.append(path)
            if device == 'cuda':
                del image
                torch.cuda.empty_cache()

    auprc_list = []
    fpr_list = []
    t_vec = np.concatenate((np.array((0.5,0.75,1.1,1.0)), np.exp(np.linspace(np.log(0.1), np.log(50), 20))))    
    for i,t in enumerate(t_vec):
        current_t_scores = []
        
        # Rileggi i logit dal disco uno alla volta
        for path in valid_paths:
            filename = os.path.basename(path)
            logits = np.load(f'temp_logits/{filename}.npy')
            
            # Applica il softmax con la temperatura t corrente
            # Usiamo torch per velocità anche se i dati sono numpy
            logits_t = torch.from_numpy(logits) / t
            probs_tensor = torch.nn.functional.softmax(logits_t, dim=0)
            
            anomaly_result_softmax = (1.0 - torch.max(probs_tensor, dim=0)[0]).numpy()
            current_t_scores.append(anomaly_result_softmax)
            
            del logits_t, probs_tensor, logits
        # Ora hai in RAM solo GLI SCORE per UNA temperatura alla volta
        [prc_auc, fpr] = eval_score(ood_gts_list, current_t_scores)
        auprc_list.append(prc_auc)
        fpr_list.append(fpr)
        if i <= 3:
            file.write(f'\n  t = {t} -->  ' + 'AUPRC softmax score:' + str(prc_auc*100.0) + '   FPR@TPR95 softmax:' + str(fpr*100.0))
            print(f't = {t}')
            print(f'AUPRC softmax score: {prc_auc*100.0}')
            print(f'FPR@TPR95 softmax: {fpr*100.0}')

        del current_t_scores
        if device == 'cuda':
            torch.cuda.empty_cache()

    performance_array = np.array(auprc_list) - np.array(fpr_list) + 1.0
    best_index = np.argmax(performance_array)
    best_t = t_vec[best_index]
    best_auprc = auprc_list[best_index]
    best_fpr = fpr_list[best_index]
    file.write(f'\n best t = {best_t} -->  ' + 'AUPRC softmax score:' + str(best_auprc*100.0) + '   FPR@TPR95 softmax:' + str(best_fpr*100.0))
    file.close()
    shutil.rmtree('temp_logits')


if __name__ == '__main__':
    main()
