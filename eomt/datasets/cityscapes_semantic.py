# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from pathlib import Path
from typing import Union
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes

from datasets.lightning_data_module import LightningDataModule
from datasets.dataset import Dataset
from datasets.transforms import Transforms


class CityscapesSemantic(LightningDataModule):
    """
    DataModule Lightning per Cityscapes in segmentazione semantica.

    Costruisce dataset train e validation leggendo gli archivi ufficiali
    Cityscapes, converte le label `id` nei rispettivi `train_id` e applica le
    trasformazioni di training quando necessario.
    """
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (1024, 1024),
        num_classes: int = 19,
        color_jitter_enabled=True,
        scale_range=(0.5, 2.0),
        check_empty_targets=True,
    ) -> None:
        """
        Inizializza il DataModule Cityscapes semantico.

        Args:
            path: Cartella contenente gli zip Cityscapes.
            num_workers: Numero di worker del DataLoader.
            batch_size: Dimensione del batch.
            img_size: Dimensione finale delle immagini.
            num_classes: Numero di classi semantiche Cityscapes.
            color_jitter_enabled: Abilita il jitter cromatico in training.
            scale_range: Intervallo di scala per lo scale jitter.
            check_empty_targets: Scarta immagini con target vuoti.
        """
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    @staticmethod
    def target_parser(target, **kwargs):
        """
        Converte una label Cityscapes in maschere binarie per classe.

        Args:
            target: Maschera label Cityscapes con id originali.
            **kwargs: Parametri aggiuntivi non usati.

        Returns:
            Tupla `(masks, labels, is_crowd)` con maschere per classe valida,
            train id corrispondenti e flag crowd sempre falsi.
        """
        masks, labels = [], []

        for label_id in target[0].unique():
            cls = next((cls for cls in Cityscapes.classes if cls.id == label_id), None)

            if cls is None or cls.ignore_in_eval:
                continue

            masks.append(target[0] == label_id)
            labels.append(cls.train_id)

        return masks, labels, [False for _ in range(len(masks))]

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        """
        Crea dataset di training e validazione per Cityscapes.

        Args:
            stage: Stage Lightning opzionale.

        Returns:
            L'istanza corrente del DataModule.
        """
        cityscapes_dataset_kwargs = {
            "img_suffix": ".png",
            "target_suffix": ".png",
            "img_stem_suffix": "leftImg8bit",
            "target_stem_suffix": "gtFine_labelIds",
            "zip_path": Path(self.path, "leftImg8bit_trainvaltest.zip"),
            "target_zip_path": Path(self.path, "gtFine_trainvaltest.zip"),
            "target_parser": self.target_parser,
            "check_empty_targets": self.check_empty_targets,
        }
        self.cityscapes_train_dataset = Dataset(
            transforms=self.transforms,
            img_folder_path_in_zip=Path("./leftImg8bit/train"),
            target_folder_path_in_zip=Path("./gtFine/train"),
            **cityscapes_dataset_kwargs,
        )
        self.cityscapes_val_dataset = Dataset(
            img_folder_path_in_zip=Path("./leftImg8bit/val"),
            target_folder_path_in_zip=Path("./gtFine/val"),
            **cityscapes_dataset_kwargs,
        )

        return self

    def train_dataloader(self):
        """
        Restituisce il DataLoader di training Cityscapes.
        """
        return DataLoader(
            self.cityscapes_train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        """
        Restituisce il DataLoader di validazione Cityscapes.
        """
        return DataLoader(
            self.cityscapes_val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )


"""class CityscapesSemanticOE(CityscapesSemantic):
    def __init__(
        self,
        path, # percorso cityscapes
        coco_root, # percorso coco
        p_ood=0.5,
        **kwargs
    ):
        super().__init__(
            path=path,
            num_classes=19,
            **kwargs
        )
        # costruisce un classico cityscapes con 19 classi

        self.coco_root = coco_root
        self.p_ood = p_ood

    def setup(self, stage=None):
        # Importiamo qui i helper OOD solo se qualcuno usa questa vecchia classe.
        from datasets.cityscapes_semantic_oe import CocoOODPaster, OODDatasetWrapper

        super().setup(stage)
        # costruisce normalmente training e validation set

        paster = CocoOODPaster(
            coco_root=self.coco_root,
            split="val2017",
            target_height_range=(80, 250), # TODO da valutare se aumentare
        )
        # Crea un oggetto che prende istanze da COCO val2017 e le incolla sulle immagini Cityscapes.

        self.cityscapes_train_dataset = OODDatasetWrapper(
            base_dataset=self.cityscapes_train_dataset,
            paster=paster,
            p_ood=self.p_ood,
        )
        # sostituisce il training dataset con un wrapper

        return self
"""