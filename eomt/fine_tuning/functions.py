import random
import cv2
import numpy as np
from pycocotools.coco import COCO
import numpy as np
import torch
from torchvision import tv_tensors
import torch.nn.functional as F

from functions import *
from eomt.functions import compute_logits, load_model
from eomt.training.mask_classification_loss import MaskClassificationLoss

# prende un oggetto dal dataset COCO e lo incolla dentro un'immagine
# viene trattato come oggetto OoD
class CocoOODPaster:
    def __init__(
        self,
        coco_root, # cartella del dataset COCO
        split="val2017", # sottoinsieme del dataset da usare
        categories=None, # categorie di COCO da cui prendere oggetti
        target_height_range=(80, 250), # intervallo per l'altezza dell'oggetto incollato
    ):
        self.coco_root = coco_root
        self.split = split
        self.img_dir = f"{coco_root}/{split}" # percorso cartella con immagini
        self.ann_file = f"{coco_root}/annotations/instances_{split}.json" # percorso cartella con annotazioni
        
        self.coco = COCO(self.ann_file) # carica le annotazioni usando pycocotools

        if categories is None:
            categories = [
                "elephant", "giraffe", "zebra", "bear",
                "couch", "chair", "toaster", "microwave",
                "banana", "apple", "backpack"
            ]

        self.categories = categories
        self.cat_ids = self.coco.getCatIds(catNms=categories)
        # converte i nomi delle categorie negli ID COCO

        self.img_ids = []
        # lista che contiene gli ID delle immagini COCO contenenti almeno una categoria scelta
        for cat_id in self.cat_ids:
            self.img_ids.extend(self.coco.getImgIds(catIds=[cat_id]))

        self.img_ids = list(set(self.img_ids))[0:300] # rimuove duplicati

        print(f'LE FOTO SCELTE DA COCO SONO{len(list(set(self.img_ids)))}')

        self.target_height_range = target_height_range

    def get_random_object(self):
    # metodo che estrae casualmente un oggetto dal dataset COCO
    
        img_id = random.choice(self.img_ids)
        # sceglie casualmente un'immagine tra quelle selezionate
        img_info = self.coco.loadImgs(img_id)[0]
        # carica le informazioni dell'immagine scelta

        ann_ids = self.coco.getAnnIds(
            imgIds=img_id,
            catIds=self.cat_ids,
            iscrowd=False
        ) # ID delle annotazioni dell'immagine scelta
        
        # ogni annotazione corrisponde ad un oggetto nell'immagine
        # contiene: ID dell'immagine a cui appartiene, categoria, bounding box

        ann = random.choice(self.coco.loadAnns(ann_ids))
        # sceglie un'annotazione casuale (oggetto da ritagliare)

        img_path = f"{self.img_dir}/{img_info['file_name']}"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = self.coco.annToMask(ann).astype(np.uint8)
        # converte l'annotazione in una maschera binaria

        ys, xs = np.where(mask > 0)
        # trova i pixel appartenenti all'oggetto
        
        # bounding box dell'oggetto
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()

        obj_img = img[ymin:ymax + 1, xmin:xmax + 1]
        # ritaglia dall'immagine originale la regione contenente l'oggetto
        obj_mask = mask[ymin:ymax + 1, xmin:xmax + 1]
        # ritaglia allo stesso modo la maschera
        
        # hanno dimensione rettangolare ma poi in paste viene
        # effettivamente incollato solo l'oggeto tramite una maschera

        cat_name = self.coco.loadCats([ann["category_id"]])[0]["name"]
        # categoria associata all'annotazione scelta

        return obj_img, obj_mask, cat_name
        # restituisce immagine ritagliata, maschera e categoria

    def resize_object(self, obj_img, obj_mask):
    # metodo che ridimensiona l'oggetto mantenendo le proprozioni
        h, w = obj_img.shape[:2] # dimensione oggetto

        target_h = random.randint(*self.target_height_range)
        # sceglie casualmente una nuova altezza nel range
        scale = target_h / h
        target_w = int(w * scale)
        # nuova larghezza mantenendo le proporzioni

        obj_img = cv2.resize(obj_img, (target_w, target_h))
        # ridimensiona l'immagine dell'oggetto
        
        obj_mask = cv2.resize(
            obj_mask,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST
        ) # ridimensiona la maschera

        return obj_img, obj_mask
        # restituisce oggetto e maschera ridimensionate

    def paste(self, city_img):
        """
        city_img: immagine RGB, array numpy H x W x 3

        returns:
            city_paste: immagine RGB con oggetto OOD incollato
            ood_mask: maschera binaria H x W
            cat_name: nome della categoria incollata
        """

        city_paste = city_img.copy()
        H, W = city_paste.shape[:2]

        obj_img, obj_mask, cat_name = self.get_random_object()
        # estrae casualmente un oggetto da COCO
        obj_img, obj_mask = self.resize_object(obj_img, obj_mask)
        # ridimensiona casualmemte l'oggetto

        h, w = obj_img.shape[:2]
        
        # Se l'oggetto è più grande dell'immagine Cityscapes, lo ridimensiona
        if w > W or h > H:
            scale = min(W / w, H / h) * 0.8

            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            obj_img = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            obj_mask = cv2.resize(obj_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            h, w = obj_img.shape[:2]
                
        # sceglie casualmente le coordinate in cui incollare l'oggetto
        x = random.randint(0, W - w)
        y = random.randint(0, H - h)

        roi = city_paste[y:y+h, x:x+w]
        # regione dell'immagine in cui verrà incollato l'oggetto

        mask_bool = obj_mask > 0 # maschera oggetto
        roi[mask_bool] = obj_img[mask_bool]
        # copia nella roi solo i pixel dell'oggetto

        city_paste[y:y+h, x:x+w] = roi
        # mette la roi nell'immagine completa

        ood_mask = np.zeros((H, W), dtype=np.uint8)
        ood_mask[y:y+h, x:x+w] = obj_mask # maschera oggetto incollato

        return city_paste, ood_mask, cat_name
    


# prende un dataset esistente
# prende ogni elemento di tale dataset e con una certa probabilità lo modifica
class OODDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, paster, p_ood=0.5):
        self.base_dataset = base_dataset # dataset originale
        self.paster = paster # oggetto che sa incollare OoD su un'immagine
        self.p_ood = p_ood # probabilità di applicare il paste

    def __len__(self):
        return len(self.base_dataset)
        # ha la stessa lunghezza del dataset originale

    def __getitem__(self, idx):
        img, target = self.base_dataset[idx]
        # prende immagine e target dal dataset originale
        # target è un dizionario con:
        # - target["masks"]    : [N, H, W] maschere binarie delle N istanze presenti
        # - target["labels"]   : [N] classe associata ad ogni istanza
        # - target["is_crowd"] : [N] flag crowd/ignore per ogni istanza
        # - target["ood_mask"] : [H, W] maschera booleana dei pixel OoD

        if random.random() > self.p_ood:
            target["ood_mask"] = torch.zeros(img.shape[-2:], dtype=torch.bool)
            # non viene applicato nessun oggetto OoD
            # restituisce una maschera tutta falsa
        else: 
            img_np = img.permute(1, 2, 0).cpu().numpy()

            # valori tra 0 e 255
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

            img, ood_mask, cat_name = self.paster.paste(img_np)

            img = torch.from_numpy(img).permute(2, 0, 1)
            img = tv_tensors.Image(img)

            target["ood_mask"] = torch.from_numpy(ood_mask > 0)
            # maschera booleana: i pixel con valore >1 diventano true
            target["ood_category"] = cat_name
            # nome della categoria incollata
        
        return img, target

def ood_hinge_loss(logits, ood_mask, alpha=5.0):
    """
    logits:   [B, 19, H, W]
    ood_mask: [B, H, W]
    """

    ood_mask = ood_mask.to(logits.device).bool()

    score = torch.tanh(logits)
    rba = -score.sum(dim=1)
    # somma le probabilità sulle 19 classi note
    # [B, 19, H, W] -> [B, H, W]

    loss_map = F.relu(alpha - rba) ** 2

    if not ood_mask.any():
        # crea un tensore con stesse caratteristiche di logits (shape, device, tiponumerico)
        loss_rba = logits.new_tensor(0.0) # se non ci sono pixel OoD nella maschera restituisce 0
    else:
        loss_rba = torch.sum(loss_map[ood_mask])
    return loss_rba


def train_one_epoch(model, train_loader, optimizer, device, lambda_oe=0.1, alpha=0.1, ignore_index=255, file=None):
    
    model.train()

    epoch_loss = 0.0
    # epoch_loss_seg = 0.0
    epoch_loss_ood = 0.0
    num_batches = 0

    mask_class_loss = MaskClassificationLoss(
                num_points = 12544,
                oversample_ratio = 3.0,
                importance_sample_ratio = 0.75,
                mask_coefficient = 5.0,
                dice_coefficient = 5.0,
                class_coefficient = 2.0,
                num_labels = 19,
                no_object_coefficient = 0.1)

    for batch_idx, batch in enumerate(train_loader):
        images, targets = batch

        optimizer.zero_grad()
        # liste per le loss del batch
        batch_losses = []
        # batch_seg_losses = []
        batch_ood_losses = []

        images = images.to(device)

        if images.dtype != torch.uint8:
            images_input = (images * 255).to(torch.uint8)
        else:
            images_input = images

        logits = compute_logits(images_input, device, model)  # [B, 19, H, W]

        # maschera semantica ID: [B, H, W], valori 0..18, ignore_index su pixel da ignorare
        masks = targets["masks"].to(device).bool()
        labels = targets["labels"].to(device).long()

        B, H, W = masks.shape
        ood_mask = targets["ood_mask"].to(device).bool()

        # semantic mask: [H, W]
        sem_mask_batch = torch.full(
            (B, H, W),
            fill_value=ignore_index, # inizialmente tutta a 255
            device=device,
            dtype=torch.long,
        )

        for m, l in zip(masks, labels):
            sem_mask_batch[m] = l # assegna ai pixel dell'oggetto la maschera corrispondente
        

        bce = F.binary_cross_entropy_with_logits(
            logits,
            masks.float(),
        )

        # AGGIUNGERE ALTRE LOSS(?): CE E DICE
        dict_losses = {"mask": }    # FORSE BASTA USARE IL FORWARD DI CLASSE MaskClassificationLoss CHE FA TUTTO IN AUTOMATICO: DA CAPIRE


            # ANCORA DA INDENTARE VISTO CHE ABBIAMO RIMOSSO IL FOR SULLE IMMAGINI
            loss_tot = mask_class_loss.loss_total()

            # loss OoD sui pixel outlier
            loss_ood = ood_hinge_loss(
                logits=logits,
                ood_mask=ood_mask,
                alpha=alpha,
            )

            loss = loss_ood #*lambda_oe   + loss_seg

            batch_losses.append(loss)
            # batch_seg_losses.append(loss_seg)
            batch_ood_losses.append(loss_ood)

        loss_batch = torch.stack(batch_losses).mean()
        #loss_seg_batch = torch.stack(batch_seg_losses).mean()
        loss_ood_batch = torch.stack(batch_ood_losses).mean()

        loss_batch.backward()
        optimizer.step()

        epoch_loss += loss_batch.item()
        #epoch_loss_seg += loss_seg_batch.item()
        epoch_loss_ood += loss_ood_batch.item()
        num_batches += 1

        if batch_idx % 20 == 0:
            msg = (
                f"batch {batch_idx:04d} | "
                f"loss={loss_batch.item():.6f} | "
                #f"loss_seg={loss_seg_batch.item():.6f} | "
                f"loss_ood={loss_ood_batch.item():.6f}"
            )

            print(msg)

            if file is not None:
                file.write(msg + "\n")
                file.flush()

    return {
        "loss": epoch_loss / max(num_batches, 1),
        #"loss_seg": epoch_loss_seg / max(num_batches, 1),
        "loss_ood": epoch_loss_ood / max(num_batches, 1),
    }


def setup_model(config, state_dict_path, device):
    model = load_model(device, config, state_dict_path)
    # Freeze backbone, unfreeze heads (come nel tuo codice)
    for param in model.parameters():
        param.requires_grad = False
    for module in [model.class_head, model.mask_head]:
        for param in module.parameters():
            param.requires_grad = True
    return model.to(device)

