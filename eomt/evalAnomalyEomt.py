# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import yaml
import glob
import torch
import random
from PIL import Image
import numpy as np
from models.eomt import EoMT
from models.vit import ViT
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
import warnings
import importlib

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
# In Cityscapes there are 19 standard classes and one more that is the OOD class
NUM_CLASSES = 20
# gpu training specific, for results' reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose([
    Resize((1024, 1024), Image.BILINEAR), 
    ToTensor()
])

target_transform = Compose([
        Resize((512, 1024), Image.NEAREST),
    ])

def load_my_state_dict(model, state_dict):
    """
    Manually loads checkpoint weights into the model.
    
    model : torch.nn.Module
        Target PyTorch model.

    state_dict : dict
        Dictionary containing the checkpoint parameters.

    Returns: torch.nn.Module, which is the model with loaded weights.
    """
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            if name.startswith("module."):
                own_state[name.split("module.")[-1]].copy_(param)
            else:
                print(name, " not loaded")
                continue
        else:
            own_state[name].copy_(param)
    return model


def extract_state_dict(checkpoint):
    """
    Extracts the model state dictionary from a checkpoint, 
    not considering all other infos, such as optimizer, 
    epoch, lr_scheduler, etc...

    Returns: dict, extracted model state dictionary, which has 
             layers' name as keys and tensors of weights as values
    """
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    if "model" in checkpoint:
        return checkpoint["model"]

    return checkpoint


def load_eomt(args, device, config=None):
    """
    Loads a pretrained EoMT model from Hugging Face.

    The function retrieves the model name either from command-line
    arguments or from the configuration file, builds the ViT-based
    EoMT architecture, downloads the pretrained weights, and loads
    them into the model.

    args : argparse.Namespace
    device : torch.device
    config : dict, optional
        Configuration dictionary used as fallback for retrieving
        the model name.

    Returns: torch.nn.Module, which is the pretrained EoMT model in evaluation mode.
    """
    name = getattr(args, "eomtName", None) # which is cityscapes_semantic_eomt_large_1024

    if name is None and config is not None:
        name = (
            config.get("trainer", {})
            .get("logger", {})
            .get("init_args", {})
            .get("name")
        )

    if name is None:
        raise ValueError(
            "Nome modello EoMT mancante. Passa --eomtName oppure mettilo nel config."
        )

    print("Loading EoMT weights from Hugging Face:", name)

    encoder = ViT(
        img_size=(1024, 1024),
        patch_size=14,
        backbone_name="vit_large_patch14_reg4_dinov2",
    )

    model = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=100, 
        num_blocks=3, # 3 layers transformer
        masked_attn_enabled=True, # attention limited to the most relevant regions
    ).to(device)
    
    state_dict_path = "/content/drive/MyDrive/Colab Notebooks/eomt_cityscapes.bin"

    checkpoint = torch.load(
        state_dict_path,
        map_location=device,
    )
    checkpoint = extract_state_dict(checkpoint)
    model = load_my_state_dict(model, checkpoint)

    model.eval()

    print("EoMT loaded successfully")

    return model

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObstacle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../models/")
    parser.add_argument('--loadModel', default="eomt.py")
    parser.add_argument('--subset', default="val")  # can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument("--eomtName", default="cityscapes_semantic_eomt_large_1024")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    # parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_eomt(args, device)

    anomaly_score_logit_list = []
    anomaly_score_softmax_list = []
    anomaly_score_entropy_list = []
    anomaly_score_rba_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)
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

        # Output of the model
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

        anomaly_result_logit = 1.0 - np.max(pixel_logits.data.to(device).numpy(), axis=0)
        probs_tensor = torch.nn.functional.softmax(pixel_logits.data.to(device), dim=0)
        anomaly_result_softmax = 1.0 - np.max(probs_tensor.numpy(), axis=0)
        anomaly_result_entropy = -torch.sum(probs_tensor * torch.log(probs_tensor), dim=0).data.to(device).numpy()            
        pathGT = path.replace("images", "labels_masks")    
        anomaly_result_rba = - torch.sum( torch.tanh(pixel_logits.data.to(device)), dim = 0) 
                    
        
        if "RoadObstacle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        # ground truth mask
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        # Discard images without anomalies
        if 1 not in np.unique(ood_gts):
            continue           
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_logit_list.append(anomaly_result_logit)
             anomaly_score_softmax_list.append(anomaly_result_softmax)
             anomaly_score_entropy_list.append(anomaly_result_entropy)
             anomaly_score_rba_list.append(anomaly_result_rba)
        del result, anomaly_result_logit, anomaly_result_softmax, anomaly_result_entropy ,ood_gts, mask
        torch.device.empty_cache()

    file.write( "\n")

    def eval_score(ood_gts_list, anomaly_score_list):
    
        ood_gts = np.array(ood_gts_list)
        anomaly_scores = np.array(anomaly_score_list)

        ood_mask = (ood_gts == 1)
        ind_mask = (ood_gts == 0)

        ood_out = anomaly_scores[ood_mask]
        ind_out = anomaly_scores[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))

        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)

        return [prc_auc, fpr]

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
