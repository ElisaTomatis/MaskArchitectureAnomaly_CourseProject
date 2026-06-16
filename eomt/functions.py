import torch
import numpy as np
import importlib

from torch.nn import functional as F
from torch.amp.autocast_mode import autocast
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score


def load_model(device, config, state_dict_path):
    """
    Costruisce e carica un modello EoMT a partire dalla configurazione e dai
    pesi salvati.

    Ricrea dinamicamente encoder, network e Lightning module usando i percorsi
    delle classi presenti in `config`, imposta il modello in modalita di
    valutazione e carica lo state dict sul dispositivo scelto.

    Args:
        device: Dispositivo su cui caricare il modello, ad esempio "cpu" o
            "cuda".
        config: Configurazione del modello in formato dizionario.
        state_dict_path: Percorso del file contenente i pesi del modello.

    Returns:
        Modello EoMT inizializzato, con pesi caricati e spostato su `device`.
    """
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

    if str(device) == "cpu":
      state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    else:
      state_dict = torch.load(state_dict_path, map_location="cuda:0", weights_only=True)
        
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    print('Model\'s weights loaded succesfully')

    return model

def compute_logits(imgs, device, model, train = False):
    """
    Calcola le logits semantiche per una o più immagini.

    Le immagini vengono spostate sul dispositivo indicato, suddivise nelle
    finestre usate da EoMT, processate dal modello e successivamente
    ricomposte nella risoluzione originale. Le logits finali rappresentano il
    punteggio di ogni classe nota per ciascun pixel.

    Se `train=True`, il forward viene eseguito all'interno di `torch.no_grad()`
    e con mixed precision (`autocast`). Se `train=False`, il forward viene
    eseguito senza questi contesti, permettendo la costruzione del grafo dei
    gradienti.

    Args:
        imgs: Lista di tensori immagine in formato compatibile con il modello.
        device: Dispositivo su cui eseguire l'inferenza o il forward.
        model: Modello EoMT già caricato.
        train: Flag che controlla l'utilizzo di `torch.no_grad()` e
            `autocast`. Default: False.

    Returns:
        Tensore delle logits per-pixel della prima immagine elaborata, con
        shape `[num_classes, H, W]`.
    """
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
    """
    Ricava il percorso della maschera ground truth associata a un'immagine.

    Sostituisce la cartella delle immagini con quella delle label e adatta
    l'estensione del file nei dataset che usano formati diversi per input e
    maschere.

    Args:
        path: Percorso dell'immagine di input.

    Returns:
        Percorso atteso della rispettiva maschera ground truth.
    """
    pathGT = path.replace("images", "labels_masks")                
    if "RoadObsticle21" in pathGT:
        pathGT = pathGT.replace("webp", "png")
    if "fs_static" in pathGT:
        pathGT = pathGT.replace("jpg", "png")                
    if "RoadAnomaly" in pathGT:
        pathGT = pathGT.replace("jpg", "png") 
    return pathGT 

def create_oodgts(mask, pathGT):
    """
    Converte una maschera ground truth nel formato binario ID/OoD.

    In base al dataset, i valori originali della maschera vengono rimappati in
    0 per pixel in-distribution, 1 per pixel out-of-distribution e 255 per pixel
    da ignorare nella valutazione.

    Args:
        mask: Maschera ground truth letta da file.
        pathGT: Percorso della maschera, usato per riconoscere il dataset.

    Returns:
        Array NumPy con la maschera rimappata in formato ID/OoD.
    """
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
    """
    Calcola le metriche principali per la valutazione dell'anomaly detection.

    Le maschere ground truth e le mappe di anomaly score vengono convertite in
    vettori di pixel ID e OoD. Su questi vettori vengono poi calcolati Average
    Precision e FPR al 95% di TPR.

    Args:
        ood_gts_list: Lista o array di maschere ground truth binarie, dove 1
            indica pixel OoD e 0 indica pixel ID.
        anomaly_score_list: Lista o array di mappe di punteggio anomalia
            associate alle immagini valutate.

    Returns:
        Lista `[prc_auc, fpr]` con Average Precision e FPR@95TPR.
    """
    
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

