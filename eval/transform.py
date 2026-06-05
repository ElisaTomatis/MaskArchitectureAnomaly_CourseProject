# Code with transformations for Cityscapes (adapted from bodokaiser/piwise code)
# Sept 2017
# Eduardo Romera
#######################

import numpy as np
import torch

from PIL import Image

def colormap_cityscapes(n):
    """
    Crea la colormap ufficiale delle classi Cityscapes.

    Restituisce un array `n x 3` di valori RGB, dove ogni riga rappresenta il
    colore associato a una classe semantica Cityscapes. Le prime 20 posizioni
    sono impostate esplicitamente secondo la palette usata per visualizzare le
    predizioni di segmentazione.
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
    Genera una colormap generica per `n` classi.

    Usa una codifica bit-wise degli indici di classe per produrre colori RGB distinti
    E' una strategia comune nei dataset di segmentazione per assegnare
    colori riproducibili a molte classi senza definirli manualmente.
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
    Trasformazione per sostituire un'etichetta con un'altra.

    Viene usata sulle maschere di segmentazione per rimappare valori specifici,
    ad esempio convertire il valore `255` della classe ignore in un indice di
    classe gestibile dal modello o dalla metrica.
    """

    def __init__(self, olabel, nlabel):
        """
        Inizializza la trasformazione di relabeling.

        `olabel` e' il valore originale da sostituire,
        `nlabel` e' il nuovo valore assegnato ai pixel corrispondenti.
        """
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        """
        Applica la sostituzione delle label al tensore ricevuto.

        La funzione richiede un tensore di tipo `LongTensor` o `ByteTensor` e
        modifica in-place tutti i pixel uguali a `olabel`, assegnando loro `nlabel`.
        """
        assert isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor) , 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class ToLabel:
    """
    Trasformazione che converte una maschera PIL in tensore di label.

    A differenza di `ToTensor`, non normalizza i valori in `[0, 1]`: conserva gli
    indici di classe come interi, formato necessario per metriche e loss di segmentazione.
    """

    def __call__(self, image):
        """
        Converte l'immagine PIL in tensore `LongTensor` con un canale.

        L'output ha forma `1 x H x W`, dove ogni valore rappresenta l'indice di
        classe del pixel corrispondente.
        """
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


class Colorize:
    """
    Trasformazione per convertire una mappa di label in immagine RGB colorata.

    Usa la colormap Cityscapes per assegnare un colore a ogni classe. E' utile
    per visualizzare predizioni o ground truth di semantic segmentation in modo leggibile.
    """

    def __init__(self, n=22):
        """
        Inizializza la trasformazione di colorizzazione.

        `n` indica il numero di classi da visualizzare. La colormap viene
        tagliata alle prime `n` classi e convertita in tensore PyTorch.
        """
        #self.cmap = colormap(256)
        self.cmap = colormap_cityscapes(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        """
        Applica la colormap a una mappa di label.

        Riceve un tensore `1 x H x W` con indici di classe e restituisce un
        tensore `3 x H x W` di tipo byte, dove ogni canale contiene i valori RGB
        associati alle classi presenti.
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
