# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import Optional
import torch
import lightning


class LightningDataModule(lightning.LightningDataModule):
    """
    Classe base per i DataModule Lightning del progetto.

    Salva parametri comuni come path, dimensione immagine, numero classi e
    opzioni del DataLoader, e fornisce funzioni di collate condivise da dataset
    semantic, instance e panoptic.
    """
    def __init__(
        self,
        path,
        batch_size: int,
        num_workers: int,
        img_size: tuple[int, int],
        num_classes: int,
        check_empty_targets: bool,
        ignore_idx: Optional[int] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        """
        Inizializza i parametri comuni dei DataModule.

        Args:
            path: Percorso principale del dataset.
            batch_size: Numero di campioni per batch.
            num_workers: Numero di worker del DataLoader.
            img_size: Dimensione delle immagini usata dalle trasformazioni.
            num_classes: Numero di classi del task.
            check_empty_targets: Se `True`, scarta target senza annotazioni.
            ignore_idx: Indice opzionale da ignorare nelle label.
            pin_memory: Abilita `pin_memory` nel DataLoader.
            persistent_workers: Mantiene i worker attivi tra epoche quando
                `num_workers > 0`.
        """
        super().__init__()

        self.path = path
        self.check_empty_targets = check_empty_targets
        self.ignore_idx = ignore_idx
        self.img_size = img_size
        self.num_classes = num_classes

        self.dataloader_kwargs = {
            "persistent_workers": False if num_workers == 0 else persistent_workers,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "batch_size": batch_size,
        }

    @staticmethod
    def train_collate(batch):
        """
        Collate function per il training.

        Args:
            batch: Lista di tuple `(img, target)`.

        Returns:
            Tupla con immagini impilate in un tensore e lista dei target.
        """
        imgs, targets = [], []

        for img, target in batch:
            imgs.append(img)
            targets.append(target)

        return torch.stack(imgs), targets

    @staticmethod
    def eval_collate(batch):
        """
        Collate function per validazione e test.

        Args:
            batch: Lista di campioni del dataset.

        Returns:
            Tuple ottenute con `zip`, mantenendo separati i campi originali.
        """
        return tuple(zip(*batch))
