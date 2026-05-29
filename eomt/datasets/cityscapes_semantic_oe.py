from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision import tv_tensors

from datasets.cityscapes_semantic import CityscapesSemantic


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
        p_ood: float = 0.2,
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
        coco_root: Optional[str | Path] = None,
        p_ood: float = 0.2,
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

    def _resolve_coco_root(self) -> Path:
        """
        Risolve la cartella COCO da argomento esplicito o variabile d'ambiente.
        """

        if self.coco_root is not None:
            return Path(self.coco_root)

        env_root = os.environ.get("EOMT_COCO_ROOT")
        if env_root:
            return Path(env_root)

        raise ValueError(
            "COCO root non specificata. Passa --data.init_args.coco_root oppure "
            "imposta la variabile d'ambiente EOMT_COCO_ROOT."
        )

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
                coco_root=self._resolve_coco_root(),
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
