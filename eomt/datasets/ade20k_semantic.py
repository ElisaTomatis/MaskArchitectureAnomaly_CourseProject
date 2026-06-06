# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from pathlib import Path
from typing import Union
from torch.utils.data import DataLoader

from datasets.lightning_data_module import LightningDataModule
from datasets.dataset import Dataset
from datasets.transforms import Transforms

CLASS_MAPPING = {i: i - 1 for i in range(1, 151)}


class ADE20KSemantic(LightningDataModule):
    """
    DataModule Lightning per ADE20K in segmentazione semantica.

    Legge immagini e annotazioni dall'archivio ADEChallengeData2016, rimappa le
    classi ADE20K da indice 1-150 a indice 0-149 e costruisce DataLoader per
    training e validazione.
    """
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (512, 512),
        num_classes: int = 150,
        color_jitter_enabled=True,
        scale_range=(0.5, 2.0),
        check_empty_targets=True,
    ) -> None:
        """
        Inizializza il DataModule semantico ADE20K.

        Args:
            path: Cartella contenente `ADEChallengeData2016.zip`.
            num_workers: Numero di worker del DataLoader.
            batch_size: Dimensione del batch.
            img_size: Dimensione finale delle immagini.
            num_classes: Numero di classi ADE20K.
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
        Converte una maschera ADE20K in maschere binarie per classe.

        Args:
            target: Maschera label ADE20K.
            **kwargs: Parametri aggiuntivi non usati.

        Returns:
            Tupla `(masks, labels, is_crowd)` con maschere per classe valida,
            label rimappate e flag crowd sempre falsi.
        """
        masks, labels = [], []

        for label_id in target[0].unique():
            cls_id = label_id.item()

            if cls_id not in CLASS_MAPPING:
                continue

            masks.append(target[0] == label_id)
            labels.append(CLASS_MAPPING[cls_id])

        return masks, labels, [False for _ in range(len(masks))]

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        """
        Crea dataset ADE20K di training e validazione.

        Args:
            stage: Stage Lightning opzionale.

        Returns:
            L'istanza corrente del DataModule.
        """
        dataset_kwargs = {
            "img_suffix": ".jpg",
            "target_suffix": ".png",
            "zip_path": Path(self.path, "ADEChallengeData2016.zip"),
            "target_zip_path": Path(self.path, "ADEChallengeData2016.zip"),
            "target_parser": self.target_parser,
            "check_empty_targets": self.check_empty_targets,
        }
        self.train_dataset = Dataset(
            img_folder_path_in_zip=Path("./ADEChallengeData2016/images/training"),
            target_folder_path_in_zip=Path(
                "./ADEChallengeData2016/annotations/training"
            ),
            transforms=self.transforms,
            **dataset_kwargs,
        )
        self.val_dataset = Dataset(
            img_folder_path_in_zip=Path("./ADEChallengeData2016/images/validation"),
            target_folder_path_in_zip=Path(
                "./ADEChallengeData2016/annotations/validation"
            ),
            **dataset_kwargs,
        )

        return self

    def train_dataloader(self):
        """
        Restituisce il DataLoader di training ADE20K.
        """
        dataset = self.train_dataset

        return DataLoader(
            dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        """
        Restituisce il DataLoader di validazione ADE20K.
        """
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
