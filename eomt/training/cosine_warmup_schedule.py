# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import math

from torch.optim.lr_scheduler import LRScheduler


class CosineWarmupSchedule(LRScheduler):
    """Scheduler step-based con warmup lineare seguito da cosine decay.

    Questo scheduler e' pensato per il fine tuning su poche epoche: per i primi
    `warmup_ratio` degli step totali aumenta linearmente il learning rate da 0
    al valore configurato nell'optimizer; per gli step rimanenti applica un
    decadimento coseno fino a `min_lr_ratio * base_lr`.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_ratio: float = 0.1,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        """Memorizza i parametri dello scheduler e inizializza LRScheduler.

        Args:
            optimizer: Optimizer PyTorch di cui modulare i learning rate.
            total_steps: Numero totale stimato di update dell'optimizer. In
                Lightning coincide con `trainer.estimated_stepping_batches`,
                quindi tiene conto di epoche, batch, gradient accumulation e
                limiti sui batch.
            warmup_ratio: Frazione degli step totali dedicata al warmup. Il
                valore predefinito 0.1 corrisponde al 10% richiesto per il fine
                tuning.
            min_lr_ratio: Frazione minima del learning rate base raggiunta alla
                fine del cosine decay. Con 0.0 il learning rate arriva a zero.
            last_epoch: Indice dell'ultimo step gestito da PyTorch; lasciarlo a
                -1 consente al framework di inizializzare correttamente lo stato.
        """

        if total_steps <= 0:
            total_steps = 1
        if not 0.0 <= warmup_ratio < 1.0:
            raise ValueError("warmup_ratio deve essere nel range [0.0, 1.0).")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio deve essere nel range [0.0, 1.0].")

        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Restituisce i learning rate del current step per ogni param group."""

        # `last_epoch` in LRScheduler rappresenta in pratica lo step corrente
        # quando lo scheduler viene chiamato a ogni update dell'optimizer.
        step = min(max(0, self.last_epoch), self.total_steps)

        if self.warmup_steps > 0 and step < self.warmup_steps:
            lr_scale = (step + 1) / self.warmup_steps
        else:
            decay_steps = max(1, self.total_steps - self.warmup_steps)
            progress = (step - self.warmup_steps) / decay_steps
            progress = min(max(progress, 0.0), 1.0)
            cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr_scale = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_scale

        return [base_lr * lr_scale for base_lr in self.base_lrs]
