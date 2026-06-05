import numpy as np
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score

def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
    """
        Carica in modo parziale i pesi di un modello PyTorch.

        La funzione confronta le chiavi presenti nello `state_dict` ricevuto con
        quelle del modello corrente. Se una chiave coincide, il relativo tensore
        viene copiato nel modello. Se la chiave non coincide ma proviene da un
        modello salvato con `DataParallel`, quindi inizia con `module.`, il
        prefisso viene rimosso e il peso viene comunque caricato nella chiave
        corrispondente. Le chiavi non riconosciute vengono ignorate e stampate.

        Questa utility e' utile quando si vuole riutilizzare un checkpoint anche
        se l'architettura locale non ha esattamente tutte le stesse chiavi del
        dizionario dei pesi salvato.
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

def eval_score(ood_gts_list, anomaly_score_list):
    """
    Calcola le metriche principali per la valutazione anomaly/OoD.

    Riceve una lista di ground truth binarie, dove 1 indica pixel anomali/OoD e
    0 indica pixel in-distribution, e una lista di score di anomalia prodotti
    dal modello o dal metodo post-hoc. I valori vengono separati in pixel OoD e
    pixel in-distribution, poi ricombinati per calcolare:

    - AuPRC, tramite `average_precision_score`, che misura la qualita' della
        classificazione pixel-wise delle anomalie.
    - FPR95, tramite `fpr_at_95_tpr`, cioe' il false positive rate quando il
        true positive rate e' fissato al 95%.

    Restituisce una lista `[prc_auc, fpr]`.
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

def create_pathGT(path):
    """
    Ricava il percorso della maschera ground truth a partire dal percorso immagine.

    I dataset usati per anomaly segmentation hanno convenzioni diverse per
    cartelle ed estensioni. La funzione sostituisce la directory `images` con
    `labels_masks` e, quando necessario, converte l'estensione dell'immagine in
    `.png` per i dataset RoadObstacle21, Fishyscapes Static e RoadAnomaly.

    Restituisce il percorso della maschera da caricare per la valutazione.
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
    Converte una maschera dataset-specifica in una maschera binaria OoD.

    La funzione prende la maschera ground truth originale e normalizza le classi
    secondo le convenzioni del dataset identificato nel percorso `pathGT`:

    - RoadAnomaly: la classe 2 viene trasformata in anomalia, cioe' 1.
    - LostAndFound: il background non valido viene messo a 255, la strada a 0 e
        gli oggetti anomali nel range previsto vengono mappati a 1.
    - Streethazard: la classe 14 viene temporaneamente marcata come 255, le
        classi note vengono poste a 0 e i pixel marcati vengono convertiti in 1.

    Il risultato e' una maschera in cui 1 indica OoD/anomalia, 0 indica
    in-distribution e 255 puo' rappresentare pixel da ignorare a seconda del
    dataset.
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