# Code with transformations for Cityscapes (adapted from bodokaiser/piwise code)
# Sept 2017
# Eduardo Romera
#######################

import numpy as np
import torch

from PIL import Image

def colormap_cityscapes(n):
    """
    Crea la palette RGB delle classi Cityscapes.

    I primi 20 colori corrispondono alle classi Cityscapes usate nella
    segmentazione semantica; eventuali righe aggiuntive restano inizializzate a
    zero.

    Args:
        n: Numero di colori da includere nella colormap.

    Returns:
        Array NumPy `[n, 3]` con valori RGB uint8.
    """
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0,:] = np.array([128, 64,128])
    cmap[1,:] = np.array([244, 35,232])
    cmap[2,:] = np.array([ 70, 70, 70])
    cmap[3,:] = np.array([ 102,102,156])
    cmap[4,:] = np.array([ 190,153,153])
    cmap[5,:] = np.array([ 153,153,153])

    cmap[6,:] = np.array([ 250,170, 30])
    cmap[7,:] = np.array([ 220,220,  0])
    cmap[8,:] = np.array([ 107,142, 35])
    cmap[9,:] = np.array([ 152,251,152])
    cmap[10,:] = np.array([ 70,130,180])

    cmap[11,:] = np.array([ 220, 20, 60])
    cmap[12,:] = np.array([ 255,  0,  0])
    cmap[13,:] = np.array([ 0,  0,142])
    cmap[14,:] = np.array([  0,  0, 70])
    cmap[15,:] = np.array([  0, 60,100])

    cmap[16,:] = np.array([  0, 80,100])
    cmap[17,:] = np.array([  0,  0,230])
    cmap[18,:] = np.array([ 119, 11, 32])
    cmap[19,:] = np.array([ 0,  0,  0])
    
    return cmap


def colormap(n):
    """
    Genera una colormap generica tramite codifica bitwise degli indici.

    Questa funzione costruisce colori distinti per classi diverse distribuendo
    i bit dell'indice sui canali RGB, una tecnica comune nelle visualizzazioni
    di segmentazione.

    Args:
        n: Numero di colori da generare.

    Returns:
        Array NumPy `[n, 3]` con valori RGB uint8.
    """
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap

class Relabel:
    """
    Trasformazione che sostituisce un'etichetta con un nuovo valore.

    Viene usata, ad esempio, per rimappare l'etichetta `255` di ignore al valore
    della classe ignore usata durante la valutazione IoU.
    """

    def __init__(self, olabel, nlabel):
        """
        Salva vecchia e nuova etichetta per la rimappatura.

        Args:
            olabel: Valore originale da sostituire.
            nlabel: Nuovo valore da assegnare.
        """
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        """
        Applica la rimappatura a un tensore di label.

        Args:
            tensor: Tensore `LongTensor` o `ByteTensor` contenente indici di
                classe.

        Returns:
            Lo stesso tensore, modificato in-place con la nuova etichetta.
        """
        assert isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor) , 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class ToLabel:
    """
    Trasformazione che converte una label PIL in tensore di indici di classe.
    """

    def __call__(self, image):
        """
        Converte una immagine di label in tensore `[1, H, W]`.

        Args:
            image: Immagine PIL contenente indici di classe.

        Returns:
            Tensore `long` con una dimensione canale aggiunta.
        """
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


class Colorize:
    """
    Trasformazione che converte una maschera di classi in immagine RGB.

    Usa la colormap Cityscapes per associare a ogni indice di classe il colore
    corrispondente, utile per visualizzare predizioni o ground truth.
    """

    def __init__(self, n=22):
        """
        Prepara la colormap da usare per la colorizzazione.

        Args:
            n: Numero di classi/colori da mantenere nella palette.
        """
        #self.cmap = colormap(256)
        self.cmap = colormap_cityscapes(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        """
        Applica la colormap a una maschera di label.

        Args:
            gray_image: Tensore `[1, H, W]` contenente indici di classe.

        Returns:
            Tensore RGB `ByteTensor` con shape `[3, H, W]`.
        """
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        #for label in range(1, len(self.cmap)):
        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
