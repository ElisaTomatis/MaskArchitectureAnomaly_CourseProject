from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from training.mask_classification_semantic import MaskClassificationSemantic


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
        rba_reduction: str = "sum",
        freeze_heads_only: bool = True,
        **kwargs,
    ) -> None:
        """Inizializza la variante OE e, se richiesto, congela tutto tranne gli head."""

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

        # La loss finale e' quella effettivamente ottimizzata: le tre loss
        # standard EoMT gia' pesate piu' la RbA pesata da lambda_rba.
        total_loss = seg_loss + self.lambda_rba * rba_loss
        self.log("losses/train_rba_loss", rba_loss, sync_dist=True, prog_bar=True)
        self.log("losses/train_loss_total_oe", total_loss, sync_dist=True, prog_bar=True)
        # Questi log hanno un solo punto per epoca: Lightning media i valori
        # scalari osservati sui batch dell'epoca, producendo grafici piu'
        # leggibili per il confronto tra iperparametri su WandB.
        self.log(
            "losses_epoch/train_loss_total_oe",
            total_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "losses_epoch/train_rba_loss",
            rba_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "losses_epoch/train_loss_without_rba",
            seg_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return total_loss
