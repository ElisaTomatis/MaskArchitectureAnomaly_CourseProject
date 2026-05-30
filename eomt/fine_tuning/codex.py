"""
Bozza guidata per il fine-tuning EoMT con Outlier Exposure.

Questo file serve come appunto leggibile e come ponte per spostare il codice
nei punti giusti del progetto.

Le parti sono organizzate cosi':
- CocoOODPaster, OODDatasetWrapper, CityscapesSemanticOE
  da mettere in `datasets/cityscapes_semantic_oe.py`
- rba_hinge_loss, MaskClassificationSemanticOE
  da mettere in `training/mask_classification_semantic_oe.py`
- OE_CONFIG_YAML
  da usare come base per un nuovo file di config in `configs/...`

Flusso desiderato:
1. `main.py` continua a usare LightningCLI e `Trainer.fit`.
2. Il datamodule produce batch Cityscapes e, con probabilita' `p_ood`, incolla
   oggetti COCO nel train set.
3. Il LightningModule calcola la loss di segmentazione EoMT + la loss RbA.
4. Si allenano solo `network.class_head` e `network.mask_head`.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from torch.optim import AdamW
from torchvision import tv_tensors

from datasets.cityscapes_semantic import CityscapesSemantic
from training.mask_classification_semantic import MaskClassificationSemantic


class CocoOODPaster:
    """
    Cosa fa:
        Estrae oggetti da COCO, li ridimensiona e li incolla su una immagine
        Cityscapes per simulare un'anomalia out-of-distribution.

    Input:
        - coco_root: directory radice di COCO
        - split: split COCO da usare, per esempio 'val2017'
        - categories: lista di categorie COCO ammesse come OOD
        - target_height_range: intervallo di altezza usato nel resize
        - max_images: limite sulle immagini COCO da considerare

    Output:
        Un oggetto con:
        - `get_random_object()` -> crop RGB, mask binaria, nome categoria
        - `resize_object()` -> oggetto e mask ridimensionati
        - `paste()` -> immagine modificata, ood_mask e nome categoria
    """

    def __init__(
        self,
        coco_root: str | Path,
        split: str = "val2017",
        categories: Optional[Sequence[str]] = None,
        target_height_range: tuple[int, int] = (80, 250),
        max_images: int = 300,
    ) -> None:
        # Normalizziamo sempre il path per evitare ambiguita' tra stringhe e Path.
        self.coco_root = Path(coco_root)
        self.split = split
        self.img_dir = self.coco_root / split
        self.ann_file = self.coco_root / "annotations" / f"instances_{split}.json"
        self.target_height_range = target_height_range

        if categories is None:
            # Categorie scelte come esempi forti di oggetti facilmente visibili.
            categories = (
                "elephant",
                "giraffe",
                "zebra",
                "bear",
                "couch",
                "chair",
                "toaster",
                "microwave",
                "banana",
                "apple",
                "backpack",
            )

        # pycocotools legge le annotation instance segmentation standard.
        self.coco = COCO(str(self.ann_file))
        self.cat_ids = self.coco.getCatIds(catNms=list(categories))

        # Prendiamo solo immagini che contengono almeno una categoria desiderata.
        img_ids: list[int] = []
        for cat_id in self.cat_ids:
            img_ids.extend(self.coco.getImgIds(catIds=[cat_id]))

        self.img_ids = list(dict.fromkeys(img_ids))[:max_images]
        if not self.img_ids:
            raise ValueError("Nessuna immagine COCO trovata per le categorie OOD.")

    def get_random_object(self, max_tries: int = 20) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Cosa fa:
            Sceglie casualmente un'istanza COCO valida e ne ritorna il crop.

        Input:
            - max_tries: massimo numero di tentativi prima di fallire

        Output:
            - obj_img: crop RGB dell'oggetto
            - obj_mask: maschera binaria ritagliata
            - cat_name: nome leggibile della categoria
        """

        for _ in range(max_tries):
            # Scegliamo una immagine e una sua annotazione casuale.
            img_id = random.choice(self.img_ids)
            img_info = self.coco.loadImgs(img_id)[0]

            ann_ids = self.coco.getAnnIds(
                imgIds=img_id,
                catIds=self.cat_ids,
                iscrowd=False,
            )
            anns = self.coco.loadAnns(ann_ids)
            if not anns:
                continue

            ann = random.choice(anns)
            # Convertiamo l'annotazione COCO in maschera binaria.
            mask = self.coco.annToMask(ann).astype(np.uint8)
            if not mask.any():
                continue

            # Troviamo il rettangolo minimo che contiene l'oggetto.
            ys, xs = np.where(mask > 0)
            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()

            # Apriamo l'immagine originale da disco e la portiamo in RGB.
            img_path = self.img_dir / img_info["file_name"]
            img = np.asarray(Image.open(img_path).convert("RGB"))

            # Crop immagine e mask sullo stesso intervallo spaziale.
            obj_img = img[ymin : ymax + 1, xmin : xmax + 1]
            obj_mask = mask[ymin : ymax + 1, xmin : xmax + 1]
            cat_name = self.coco.loadCats([ann["category_id"]])[0]["name"]
            return obj_img, obj_mask, cat_name

        raise RuntimeError("Impossibile estrarre un oggetto COCO valido.")

    def resize_object(
        self,
        obj_img: np.ndarray,
        obj_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Cosa fa:
            Ridimensiona oggetto e mask mantenendo il rapporto tra i lati.

        Input:
            - obj_img: crop RGB dell'oggetto
            - obj_mask: mask binaria dell'oggetto

        Output:
            - img: oggetto ridimensionato
            - mask: mask ridimensionata con nearest neighbor
        """

        h, w = obj_img.shape[:2]
        # Scegliamo una nuova altezza e manteniamo l'aspect ratio.
        target_h = random.randint(*self.target_height_range)
        target_w = max(1, round(w * target_h / h))

        # Bilinear per l'immagine, nearest per la mask.
        img = Image.fromarray(obj_img).resize((target_w, target_h), Image.BILINEAR)
        mask = Image.fromarray(obj_mask).resize((target_w, target_h), Image.NEAREST)
        return np.asarray(img), np.asarray(mask)

    def paste(self, city_img: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Cosa fa:
            Incolla un oggetto COCO dentro una immagine Cityscapes.

        Input:
            - city_img: immagine RGB come array numpy H x W x 3

        Output:
            - city_paste: immagine modificata con il paste
            - ood_mask: maschera binaria H x W dei pixel anomali
            - cat_name: nome della categoria incollata
        """

        city_paste = city_img.copy()
        H, W = city_paste.shape[:2]

        # Estraiamo e ridimensioniamo un oggetto COCO prima di incollarlo.
        obj_img, obj_mask, cat_name = self.get_random_object()
        obj_img, obj_mask = self.resize_object(obj_img, obj_mask)
        h, w = obj_img.shape[:2]

        if w > W or h > H:
            # Se l'oggetto e' ancora troppo grande, lo scalamo in modo conservativo.
            scale = min(W / w, H / h) * 0.8
            new_w = max(1, round(w * scale))
            new_h = max(1, round(h * scale))
            obj_img = np.asarray(
                Image.fromarray(obj_img).resize((new_w, new_h), Image.BILINEAR)
            )
            obj_mask = np.asarray(
                Image.fromarray(obj_mask).resize((new_w, new_h), Image.NEAREST)
            )
            h, w = obj_img.shape[:2]

        # Posizioniamo l'oggetto in una posizione casuale dell'immagine.
        x = random.randint(0, W - w)
        y = random.randint(0, H - h)

        mask_bool = obj_mask > 0
        roi = city_paste[y : y + h, x : x + w]
        # Sovrascriviamo solo i pixel appartenenti all'oggetto.
        roi[mask_bool] = obj_img[mask_bool]
        city_paste[y : y + h, x : x + w] = roi

        # La OOD mask segnala esattamente i pixel coperti dall'anomalia.
        ood_mask = np.zeros((H, W), dtype=np.uint8)
        ood_mask[y : y + h, x : x + w][mask_bool] = 1
        return city_paste, ood_mask, cat_name


def _clone_target(target: dict) -> dict:
    """
    Cosa fa:
        Duplica il dizionario target per evitare modifiche in-place.

    Input:
        - target: dizionario con maschere, label e flag vari

    Output:
        - cloned: copia superficiale con tensori clonati
    """

    cloned = {}
    for key, value in target.items():
        # Cloniamo solo i tensori, lasciando invariati eventuali oggetti non tensoriali.
        cloned[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return cloned


class OODDatasetWrapper(torch.utils.data.Dataset):
    """
    Cosa fa:
        Avvolge un dataset Cityscapes e, con probabilita' `p_ood`, inserisce un
        oggetto COCO OOD nella sample, aggiungendo anche `target["ood_mask"]`.

    Input:
        - base_dataset: dataset Cityscapes gia' costruito
        - paster: istanza di CocoOODPaster
        - p_ood: probabilita' di applicare il paste OOD

    Output:
        - __len__(): stessa lunghezza del dataset base
        - __getitem__(): coppia (img, target) eventualmente modificata
    """

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        paster: CocoOODPaster,
        p_ood: float = 0.5,
    ) -> None:
        self.base_dataset = base_dataset
        self.paster = paster
        self.p_ood = p_ood

    def __len__(self) -> int:
        """Restituisce la lunghezza del dataset base."""
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        """
        Cosa fa:
            Recupera una sample Cityscapes e, a volte, la trasforma in sample OOD.

        Input:
            - idx: indice della sample

        Output:
            - img: immagine originale o con oggetto COCO incollato
            - target: dizionario con masks, labels, is_crowd e ood_mask
        """

        img, target = self.base_dataset[idx]
        target = _clone_target(target)

        if random.random() > self.p_ood:
            # Nessuna anomalia: la maschera OOD resta vuota.
            target["ood_mask"] = torch.zeros(img.shape[-2:], dtype=torch.bool)
            return img, target

        # Teniamo una copia di sicurezza nel caso il paste renda invalida la sample.
        original_img = img
        original_target = _clone_target(target)

        # Convertiamo l'immagine in numpy uint8 perché il paster usa PIL/numpy.
        img_np = img.permute(1, 2, 0).cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        pasted_img, ood_mask_np, cat_name = self.paster.paste(img_np)
        ood_mask = torch.from_numpy(ood_mask_np > 0)

        # I pixel coperti dall'oggetto OOD non devono restare nella supervisione semantica.
        visible_masks = target["masks"].bool() & ~ood_mask
        valid = visible_masks.flatten(1).any(dim=1)
        if not valid.any():
            # Se il paste distrugge tutta la supervisione, torniamo alla sample originale.
            original_target["ood_mask"] = torch.zeros(
                original_img.shape[-2:], dtype=torch.bool
            )
            return original_img, original_target

        # Teniamo solo le istanze ancora visibili dopo il paste.
        target["masks"] = tv_tensors.Mask(visible_masks[valid])
        target["labels"] = target["labels"][valid]
        target["is_crowd"] = target["is_crowd"][valid]
        target["ood_mask"] = ood_mask
        target["ood_category"] = cat_name

        # Ricostruiamo il tensore immagine in formato CHW.
        img_tensor = torch.from_numpy(pasted_img).permute(2, 0, 1).contiguous()
        return tv_tensors.Image(img_tensor), target


class CityscapesSemanticOE(CityscapesSemantic):
    """
    Cosa fa:
        Estende il DataModule Cityscapes standard e sostituisce il train set con
        un wrapper che introduce immagini con anomalie OOD.

    Input:
        - path: root di Cityscapes
        - coco_root: root di COCO
        - p_ood: probabilita' di applicare il paste durante il training
        - coco_split: split COCO da cui pescare gli oggetti
        - ood_categories: categorie COCO ammesse
        - ood_target_height_range: range di resize degli oggetti COCO
        - kwargs: argomenti standard del DataModule base

    Output:
        - setup(): costruisce train/val dataset
        - train_dataloader(): dataloader del training con OOD
        - val_dataloader(): dataloader di validazione standard
    """

    def __init__(
        self,
        path,
        coco_root,
        p_ood: float = 0.5,
        coco_split: str = "val2017",
        ood_categories: Optional[Sequence[str]] = None,
        ood_target_height_range: tuple[int, int] = (80, 250),
        **kwargs,
    ) -> None:
        # Riusiamo tutta l'infrastruttura del DataModule Cityscapes standard.
        super().__init__(path=path, **kwargs)
        self.coco_root = coco_root
        self.p_ood = p_ood
        self.coco_split = coco_split
        self.ood_categories = ood_categories
        self.ood_target_height_range = ood_target_height_range

    def setup(self, stage=None):
        """
        Cosa fa:
            Crea train/val dataset Cityscapes e wrappa il train set con OOD.

        Input:
            - stage: fase Lightning, usata per capire se stiamo facendo fit

        Output:
            - self con attributi `cityscapes_train_dataset` e `cityscapes_val_dataset`
        """

        super().setup(stage)

        if stage in (None, "fit"):
            # Il paster si costruisce una sola volta durante il setup.
            paster = CocoOODPaster(
                coco_root=self.coco_root,
                split=self.coco_split,
                categories=self.ood_categories,
                target_height_range=self.ood_target_height_range,
            )
            self.cityscapes_train_dataset = OODDatasetWrapper(
                base_dataset=self.cityscapes_train_dataset,
                paster=paster,
                p_ood=self.p_ood,
            )

        return self


def rba_hinge_loss(
    per_pixel_scores: torch.Tensor,
    ood_mask: torch.Tensor,
    alpha: float = 5.0,
    reduction: str = "sum",
) -> torch.Tensor:
    """
    Cosa fa:
        Calcola la loss RbA (Rejected by All) sui pixel marcati come OOD.

    Input:
        - per_pixel_scores: tensore [B, C, H, W] con score semantici per classe
        - ood_mask: tensore [B, H, W] con True sui pixel anomali
        - alpha: margine della hinge loss
        - reduction: 'mean' oppure 'sum'

    Output:
        - loss scalare che penalizza i pixel OOD troppo "sicuri"
    """

    # Portiamo la maschera sullo stesso device dei punteggi.
    ood_mask = ood_mask.to(per_pixel_scores.device).bool()
    if not ood_mask.any():
        # Nessun pixel OOD nel batch: la loss deve essere zero.
        return per_pixel_scores.new_zeros(())

    # Tanh comprime i valori e li rende piu' stabili numericamente.
    score = torch.tanh(per_pixel_scores)
    # RbA usa il rifiuto globale come somma negativa sugli score di classe.
    rba = -score.sum(dim=1)
    # Hinge quadratica: penalizza i pixel che non oltrepassano il margine alpha.
    loss_map = F.relu(alpha - rba).pow(2)
    selected = loss_map[ood_mask]

    if reduction == "sum":
        return selected.sum()
    if reduction == "mean":
        return selected.mean()
    raise ValueError(f"Riduzione RbA non supportata: {reduction}")


class MaskClassificationSemanticOE(MaskClassificationSemantic):
    """
    Cosa fa:
        Specializza il LightningModule di semantic segmentation aggiungendo la
        loss RbA e congelando tutti i parametri tranne `class_head` e `mask_head`.

    Input:
        Riceve gli stessi argomenti di `MaskClassificationSemantic`, piu':
        - lambda_rba: peso della loss RbA
        - rba_alpha: margine della hinge
        - rba_reduction: riduzione della loss
        - freeze_heads_only: se True congela tutto tranne gli head

    Output:
        - `training_step()` restituisce la loss totale
        - `configure_optimizers()` crea un AdamW sui soli parametri trainable
    """

    def __init__(
        self,
        *args,
        lambda_rba: float = 0.1,
        rba_alpha: float = 5.0,
        rba_reduction: str = "mean",
        freeze_heads_only: bool = True,
        **kwargs,
    ) -> None:
        # Costruiamo la classe base che contiene modello, loss e metriche.
        super().__init__(*args, **kwargs)
        self.lambda_rba = lambda_rba
        self.rba_alpha = rba_alpha
        self.rba_reduction = rba_reduction
        self.freeze_heads_only = freeze_heads_only

        if freeze_heads_only:
            # Applichiamo il freeze richiesto dal progetto.
            self.freeze_all_but_heads()

    def freeze_all_but_heads(self) -> None:
        """
        Cosa fa:
            Imposta `requires_grad=False` su tutto il modello e riattiva solo
            `network.class_head` e `network.mask_head`.

        Input:
            Nessuno.

        Output:
            Nessuno, ma modifica i flag `requires_grad` dei parametri.
        """

        for param in self.network.parameters():
            param.requires_grad = False

        for module_name in ("class_head", "mask_head"):
            module = getattr(self.network, module_name)
            for param in module.parameters():
                param.requires_grad = True

        trainable = [name for name, p in self.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("Nessun parametro trainable: controlla i nomi degli head.")

    def on_fit_start(self) -> None:
        """Stampa un riepilogo dei parametri trainable quando parte il fit."""

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        self.print(f"Trainable parameters: {trainable:,} / {total:,}")

    def configure_optimizers(self):
        """
        Cosa fa:
            Costruisce l'optimizer solo sui parametri non congelati.

        Input:
            Nessuno esplicito, usa gli attributi del modulo.

        Output:
            - un AdamW sui soli parametri trainable
        """

        trainable_params = [p for p in self.parameters() if p.requires_grad]
        return AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def _extract_ood_masks(
        self,
        targets: list[dict],
        size: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Cosa fa:
            Estrae e, se serve, ridimensiona le ood_mask presenti nei target.

        Input:
            - targets: lista di dizionari target del batch
            - size: altezza/larghezza attese
            - device: device su cui portare i tensori

        Output:
            - tensore [B, H, W] di maschere booleane
        """

        masks = []
        for target in targets:
            mask = target.get("ood_mask")
            if mask is None:
                mask = torch.zeros(size, dtype=torch.bool)

            mask = mask.to(device)
            if tuple(mask.shape[-2:]) != tuple(size):
                # Allineiamo la mask alla risoluzione dell'input.
                mask = F.interpolate(
                    mask[None, None].float(),
                    size=size,
                    mode="nearest",
                )[0, 0].bool()
            masks.append(mask.bool())

        return torch.stack(masks, dim=0)

    def training_step(self, batch, batch_idx):
        """
        Cosa fa:
            Calcola la loss EoMT standard e la loss RbA sullo stesso batch.

        Input:
            - batch: coppia (imgs, targets) dal DataLoader
            - batch_idx: indice del batch corrente

        Output:
            - loss scalare totale usata da Lightning per backward
        """

        imgs, targets = batch

        # Forward EoMT: otteniamo una predizione per ciascun blocco.
        mask_logits_per_block, class_logits_per_block = self(imgs)

        losses_all_blocks = {}
        for i, (mask_logits, class_logits) in enumerate(
            zip(mask_logits_per_block, class_logits_per_block)
        ):
            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=targets,
            )
            block_postfix = self.block_postfix(i)
            # Ogni blocco contribuisce con il suo set di loss.
            losses_all_blocks |= {
                f"{key}{block_postfix}": value for key, value in losses.items()
            }

        # Loss di segmentazione standard, gia' pesata come nella repo originale.
        seg_loss = self.criterion.loss_total(losses_all_blocks, self.log)

        # Usiamo l'ultimo blocco per la parte RbA a livello pixel.
        final_mask_logits = F.interpolate(
            mask_logits_per_block[-1],
            size=imgs.shape[-2:],
            mode="bilinear",
        )
        per_pixel_scores = self.to_per_pixel_logits_semantic(
            final_mask_logits,
            class_logits_per_block[-1],
        )
        ood_masks = self._extract_ood_masks(
            targets=targets,
            size=imgs.shape[-2:],
            device=imgs.device,
        )
        rba_loss = rba_hinge_loss(
            per_pixel_scores=per_pixel_scores,
            ood_mask=ood_masks,
            alpha=self.rba_alpha,
            reduction=self.rba_reduction,
        )

        # La loss finale e' la somma delle due componenti.
        total_loss = seg_loss + self.lambda_rba * rba_loss
        self.log("losses/train_rba_loss", rba_loss, sync_dist=True, prog_bar=True)
        self.log("losses/train_loss_total_oe", total_loss, sync_dist=True, prog_bar=True)
        return total_loss


OE_CONFIG_YAML = """
trainer:
  max_epochs: 20
  logger:
    class_path: lightning.pytorch.loggers.wandb.WandbLogger
    init_args:
      resume: allow
      project: "eomt"
      name: "cityscapes_semantic_eomt_base_640_oe"

model:
  class_path: training.mask_classification_semantic_oe.MaskClassificationSemanticOE
  init_args:
    lambda_rba: 0.1
    rba_alpha: 5.0
    rba_reduction: mean
    freeze_heads_only: True
    attn_mask_annealing_enabled: True
    attn_mask_annealing_start_steps: [3317, 8292, 13268]
    attn_mask_annealing_end_steps: [6634, 11609, 16585]
    network:
      class_path: models.eomt.EoMT
      init_args:
        num_q: 100
        num_blocks: 3
        encoder:
          class_path: models.vit.ViT
          init_args:
            backbone_name: vit_base_patch14_reg4_dinov2

data:
  class_path: datasets.cityscapes_semantic_oe.CityscapesSemanticOE
  init_args:
    p_ood: 0.5
    coco_split: val2017
    ood_target_height_range: [80, 250]
"""
