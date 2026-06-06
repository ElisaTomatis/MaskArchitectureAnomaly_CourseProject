# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from torch import nn
from timm.layers import LayerNorm2d


class ScaleBlock(nn.Module):
    """
    Blocco di upscaling usato per aumentare la risoluzione delle feature.

    Combina una convoluzione trasposta 2x, un'attivazione GELU, una convoluzione
    depthwise 3x3 e una normalizzazione 2D. In EoMT viene usato per portare le
    feature patch del ViT a una risoluzione piu adatta alla predizione delle
    maschere.
    """
    def __init__(self, embed_dim, conv1_layer=nn.ConvTranspose2d):
        """
        Inizializza il blocco di upscaling.

        Args:
            embed_dim: Numero di canali delle feature in input e output.
            conv1_layer: Tipo di layer usato per il primo upscaling 2x.
        """
        super().__init__()

        self.conv1 = conv1_layer(
            embed_dim,
            embed_dim,
            kernel_size=2,
            stride=2,
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
            bias=False,
        )
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x):
        """
        Applica upscaling, attivazione, convoluzione depthwise e normalizzazione.

        Args:
            x: Tensore feature con shape `[B, C, H, W]`.

        Returns:
            Tensore feature con stessa dimensione canale e risoluzione spaziale
            raddoppiata dal primo layer.
        """
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)

        return x
