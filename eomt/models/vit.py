# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import Optional
import torch
import torch.nn as nn

import timm
from transformers import AutoModel


class ViT(nn.Module):
    """
    Wrapper per encoder Vision Transformer usato da EoMT.

    La classe carica un backbone ViT da `timm` oppure da Hugging Face
    Transformers, espone un'interfaccia compatibile con EoMT e registra media e
    deviazione standard ImageNet per la normalizzazione delle immagini.
    """
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size=16,
        backbone_name="vit_large_patch14_reg4_dinov2",
        ckpt_path: Optional[str] = None,
    ):
        """
        Inizializza il backbone ViT.

        Args:
            img_size: Dimensione delle immagini in input `(H, W)`.
            patch_size: Dimensione delle patch, usata per i modelli `timm`.
            backbone_name: Nome del backbone da caricare. Se contiene `/`,
                viene interpretato come modello Hugging Face; altrimenti come
                modello `timm`.
            ckpt_path: Percorso opzionale di un checkpoint locale. Se assente,
                il modello `timm` viene caricato con pesi pre-addestrati.
        """
        super().__init__()

        if "/" in backbone_name:
            self.backbone = self.transformers_to_timm(
                AutoModel.from_pretrained(
                    backbone_name,
                ),
                img_size,
            )
        else:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=ckpt_path is None,
                img_size=img_size,
                patch_size=patch_size,
                num_classes=0,
            )

        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def transformers_to_timm(self, backbone, img_size: tuple[int, int]):
        """
        Adatta un backbone Hugging Face alla struttura attesa dal codice `timm`.

        EoMT si aspetta attributi come `patch_embed`, `blocks`, `embed_dim` e
        `num_prefix_tokens`. Questa funzione rinomina o crea tali attributi a
        partire dal modello Transformers e rimuove i campi non piu necessari.

        Args:
            backbone: Modello ViT caricato da Hugging Face Transformers.
            img_size: Dimensione dell'immagine, usata per calcolare la griglia
                delle patch.

        Returns:
            Backbone adattato all'interfaccia usata da EoMT.
        """
        backbone.patch_embed = backbone.embeddings
        backbone.patch_embed.patch_size = (
            backbone.embeddings.config.patch_size,
            backbone.embeddings.config.patch_size,
        )
        backbone.patch_embed.grid_size = (
            img_size[0] // backbone.embeddings.config.patch_size,
            img_size[1] // backbone.embeddings.config.patch_size,
        )

        backbone.embed_dim = backbone.embeddings.config.hidden_size
        backbone.num_prefix_tokens = backbone.patch_embed.config.num_register_tokens + 1
        backbone.blocks = backbone.layer

        del (
            backbone.patch_embed.mask_token,
            backbone.embeddings,
            backbone.layer,
        )

        return backbone
