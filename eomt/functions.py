import torch
import numpy as np
import importlib

from torch.nn import functional as F
from torch.amp.autocast_mode import autocast
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score

import matplotlib.pyplot as plt


def load_model(device, config, state_dict_path):
    # Load encoder
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    encoder = encoder_cls(img_size=(1024, 1024), **encoder_cfg.get("init_args", {}))

    # Load network
    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(
        masked_attn_enabled=False,
        num_classes=19,
        encoder=encoder,
        **network_kwargs,
    )

    # Load Lightning module
    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    if "stuff_classes" in config["data"].get("init_args", {}):
        model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

    model = (
        lit_cls(
            img_size=(1024, 1024),
            num_classes=19,
            network=network,
            **model_kwargs,
        )
        .eval()
        .to(device)
    )

    if device == 'cpu':
        state_dict = torch.load(
                    state_dict_path, map_location="cpu", weights_only=True
                )
    else:
        state_dict = torch.load(
                    state_dict_path, map_location=f"cuda:{0}", weights_only=True
                )
    model.load_state_dict(state_dict, strict=False)
    print('Model\'s weights loaded succesfully')

    return model

def compute_logits(imgs, device, model, train = False):
    if train:
        with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):
            imgs = [img.to(device) for img in imgs]
            img_sizes = [img.shape[-2:] for img in imgs]
            crops, origins = model.window_imgs_semantic(imgs)

            mask_logits_per_layer, class_logits_per_layer = model(crops)
            mask_logits = F.interpolate(
                mask_logits_per_layer[-1], (1024, 1024), mode="bilinear"
            )

            crop_logits = model.to_per_pixel_logits_semantic(
                mask_logits, class_logits_per_layer[-1]
            )
            logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
        
    else:
        imgs = [img.to(device) for img in imgs]
        img_sizes = [img.shape[-2:] for img in imgs]
        crops, origins = model.window_imgs_semantic(imgs)

        mask_logits_per_layer, class_logits_per_layer = model(crops)
        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], (1024, 1024), mode="bilinear"
        )

        crop_logits = model.to_per_pixel_logits_semantic(
            mask_logits, class_logits_per_layer[-1]
        )
        logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)

    return logits[0]

def create_pathGT(path):
    pathGT = path.replace("images", "labels_masks")                
    if "RoadObsticle21" in pathGT:
        pathGT = pathGT.replace("webp", "png")
    if "fs_static" in pathGT:
        pathGT = pathGT.replace("jpg", "png")                
    if "RoadAnomaly" in pathGT:
        pathGT = pathGT.replace("jpg", "png") 
    return pathGT 

def create_oodgts(mask, pathGT):
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
    return ood_gts

def eval_score(ood_gts_list, anomaly_score_list):
    
    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)
    
    """im = plt.imshow(anomaly_scores[0,:,:], cmap='hot')
    plt.colorbar(im, label='Punteggio Anomalia')

    plt.axis('off') # Se vuoi nascondere le coordinate dei pixel
    plt.savefig(f'anomalia_{counter}.png', bbox_inches='tight', pad_inches=0)
    plt.close()"""

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

