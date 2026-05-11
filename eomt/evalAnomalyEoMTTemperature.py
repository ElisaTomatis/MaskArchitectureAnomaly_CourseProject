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
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose([
    Resize((512, 1024), Image.BILINEAR), # EoMT Giant usa solitamente 1280
    ToTensor(),
    # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # Standard ImageNet/DINO
])

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)

def load_my_state_dict(model, state_dict):
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

# serve a estrarre lo state_dict da un checkpoint
def extract_state_dict(checkpoint):
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    if "model" in checkpoint:
        return checkpoint["model"]

    return checkpoint


def load_eomt(args, device, config=None):
    # 1. Prendi il nome del modello
    name = getattr(args, "eomtName", None)

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
        img_size=(512, 1024),
        patch_size=14,
        backbone_name="vit_large_patch14_reg4_dinov2",
    )

    model = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=100, # cerca fino a 100 oggetti diversi per ogni immagine
        num_blocks=4, # usiamo gli ultimi 4 blocchi del Transformer
        masked_attn_enabled=True, # limita l'attenzione delle query solo alle regioni dove è stata inizialmente trovata una maschera
    ).to(device)
    
    # 4. Scarica pesi
    try:
        state_dict_path = hf_hub_download(
            repo_id=f"tue-mps/{name}",
            filename="pytorch_model.bin",
        )
    except RepositoryNotFoundError:
        raise RepositoryNotFoundError(
            f"Repository Hugging Face non trovato: tue-mps/{name}"
        )

    # 5. Carica pesi
    checkpoint = torch.load(
        state_dict_path,
        map_location=device,
        weights_only=True,
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
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../models/")
    parser.add_argument('--loadModel', default="eomt.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    # TODO: understand if it is needed
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument("--eomtName", default="cityscapes_semantic_eomt_large_1024")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_eomt(args, device)

    anomaly_score_softmax_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'w')

    t_vec = np.concatenate((np.array((0.5,0.75,1.1)), np.exp(np.linspace(np.log(0.1), np.log(50), 20))))    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()
        # images = images.permute(0,3,1,2)
        with torch.no_grad():
            result = model(images)
        
        mask_logits_per_layer = result[0][-1]
        class_logits_per_layer = result[1][-1]

        mask_logits_per_layer = torch.nn.functional.interpolate(
          mask_logits_per_layer,
          size=(512, 1024),
          mode="bilinear",
          align_corners=False,
        )

        mask_prob = torch.sigmoid(mask_logits_per_layer) 
        class_prob = torch.softmax(class_logits_per_layer, dim=-1)

        # B, Q, C = class_prob.shape
        _, _, H, W = mask_prob.shape
        cp = class_prob.transpose(1, 2) 
        mp = torch.flatten(input = mask_prob,start_dim = 2) 
        # This operation is performed for each batch
        pixel_logits = torch.matmul(cp, mp) 
        pixel_logits = pixel_logits.unflatten(2, (H, W))
        pixel_logits = pixel_logits.squeeze(0)
        
        anomaly_result_list = []
        for t in t_vec:
            probs_tensor = torch.nn.functional.softmax(pixel_logits.data.cpu() / t, dim=0)  
            anomaly_result_softmax = 1.0 - np.max(probs_tensor.numpy(), axis=0)
            anomaly_result_list.append(anomaly_result_softmax)    

        pathGT = path.replace("images", "labels_masks")      
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        mask = target_transform(mask)
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

        if 1 not in np.unique(ood_gts):
            continue           
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_softmax_list.append(anomaly_result_list)
        del result, anomaly_result_softmax, ood_gts, mask
        torch.cuda.empty_cache()

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
