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

"""
Valutazione anomaly detection di EoMT su dataset con maschere OoD.

Lo script carica un modello EoMT addestrato su Cityscapes, esegue inferenza
sulle immagini passate tramite `--input` e confronta diverse strategie di
anomaly score: logit, softmax, entropia e RBA. Per ogni strategia calcola AUPRC
e FPR@TPR95 usando le ground truth rimappate in formato ID/OoD.
"""

IGNORE_INDEX = 255

input_transform = Compose([
    Resize((1024, 1024), Image.BILINEAR), 
    ToTensor()
])

target_transform = Compose([
        Resize((1024, 1024), Image.NEAREST),
    ])


def main():
    """
    Esegue la valutazione anomaly detection con piu punteggi di anomalia.

    La funzione legge il pattern di immagini da linea di comando, carica
    configurazione e pesi del modello, calcola le logits per ogni immagine e
    genera quattro mappe di anomaly score: massimo logit inverso, massimo
    softmax inverso, entropia della distribuzione softmax e score RBA. Le
    immagini senza pixel OoD vengono ignorate; sulle rimanenti vengono calcolati
    AUPRC e FPR@TPR95.

    Args:
        None. Gli argomenti vengono letti da `ArgumentParser`.

    Returns:
        None. I risultati vengono stampati a schermo e scritti in
        `results.txt`.
    """
    parser = ArgumentParser()
    parser.add_argument("--input") 
    parser.add_argument("--weights_dir", default='/content/drive/MyDrive/ml_anomaly_segmentation/eomt_cityscapes.bin')   
    args = parser.parse_args()

    seed_everything(0, verbose=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    config_path = 'configs/dinov2/cityscapes/semantic/eomt_base_640.yaml' 
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    state_dict_path = args.weights_dir
    
    warnings.filterwarnings("ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )

    model = load_model(device, config, state_dict_path)
    
    if not os.path.exists('results_finetuned.txt'):
        open('results_finetuned.txt', 'w').close()
    file = open('results_finetuned.txt', 'w')

    anomaly_score_logit_list = []
    anomaly_score_softmax_list = []
    anomaly_score_entropy_list = []
    anomaly_score_rba_list = []
    ood_gts_list = []

    for path in glob.glob(os.path.expanduser(str(args.input))):
        print(path)
        if device == torch.device('cpu'):
            image = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float()
        else:
            image = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()
        image = image.squeeze(0)
        image = (image * 255).to(torch.uint8)
        pixel_logits = compute_logits([image], device, model)
        
        anomaly_result_logit = 1.0 - np.max(pixel_logits.data.cpu().numpy(), axis=0)
        probs_tensor = F.softmax(pixel_logits.data.cpu(), dim=0)
        anomaly_result_softmax = 1.0 - np.max(probs_tensor.numpy(), axis=0)
        anomaly_result_entropy = -torch.sum(probs_tensor * torch.log(probs_tensor), dim=0).data.cpu().numpy()              
        anomaly_result_rba = - torch.sum( torch.tanh(pixel_logits.data.cpu()), dim = 0).data.cpu().numpy() 
                    
        pathGT = create_pathGT(path)  
        mask = Image.open(pathGT)
        mask = target_transform(mask)
        # ground truth mask
        ood_gts = create_oodgts(mask, pathGT)

        # Discard images without anomalies
        if 1 not in np.unique(ood_gts):
            continue           
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_logit_list.append(anomaly_result_logit)
             anomaly_score_softmax_list.append(anomaly_result_softmax)
             anomaly_score_entropy_list.append(anomaly_result_entropy)
             anomaly_score_rba_list.append(anomaly_result_rba)
        del pixel_logits, anomaly_result_logit, anomaly_result_softmax, anomaly_result_entropy, anomaly_result_rba, ood_gts, mask
        if device == 'cuda':
            torch.cuda.empty_cache()

    file.write( "\n")

    [prc_auc_logit, fpr_logit] = eval_score(ood_gts_list, anomaly_score_logit_list)
    [prc_auc_softmax, fpr_softmax] = eval_score(ood_gts_list, anomaly_score_softmax_list)
    [prc_auc_entropy, fpr_entropy] = eval_score(ood_gts_list, anomaly_score_entropy_list)
    [prc_auc_rba, fpr_rba] = eval_score(ood_gts_list, anomaly_score_rba_list)

    print(f'AUPRC logit score: {prc_auc_logit*100.0}')
    print(f'FPR@TPR95 logit: {fpr_logit*100.0}')

    print(f'AUPRC softmax score: {prc_auc_softmax*100.0}')
    print(f'FPR@TPR95 softmax: {fpr_softmax*100.0}')

    print(f'AUPRC entropy score: {prc_auc_entropy*100.0}')
    print(f'FPR@TPR95 entropy: {fpr_entropy*100.0}')

    print(f'AUPRC rba score: {prc_auc_rba*100.0}')
    print(f'FPR@TPR95 rba: {fpr_rba*100.0}')


    file.write(('    AUPRC logit score:' + str(prc_auc_logit*100.0) + '   FPR@TPR95 logit:' + str(fpr_logit*100.0) +
                '\n    AUPRC softmax score:' + str(prc_auc_softmax*100.0) + '   FPR@TPR95 softmax:' + str(fpr_softmax*100.0) +
                '\n    AUPRC entropy score:' + str(prc_auc_entropy*100.0) + '   FPR@TPR95 entropy:' + str(fpr_entropy*100.0) +
                '\n    AUPRC rba score:' + str(prc_auc_rba*100.0) + '   FPR@TPR95 rba:' + str(fpr_rba*100.0)
                ))

    file.close()


if __name__ == '__main__':
    main()
