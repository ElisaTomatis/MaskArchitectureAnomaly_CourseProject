# Code with dataset loader for VOC12 and Cityscapes (adapted from bodokaiser/piwise code)
# Sept 2017
# Eduardo Romera
#######################

import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    """
    Carica un'immagine da un file usando PIL.

    Args:
        file: file path o file object da cui leggere l'immagine.

    Returns:
        Immagine PIL aperta (non convertita).
    """
    return Image.open(file)

def is_image(filename):
    """
    Controlla se un file è un'immagine valida.

    Args:
        filename: nome del file da verificare.

    Returns:
        True se il file termina con una delle estensioni in EXTENSIONS.
    """
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    """
    Verifica se un file è una label Cityscapes in formato trainId.

    Args:
        filename: nome del file da verificare.

    Returns:
        True se il file termina con '_labelTrainIds.png'.
    """
    return filename.endswith("_labelTrainIds.png")

def image_path(root, basename, extension):
    """
    Costruisce un path completo per un'immagine o label.

    Args:
        root: directory root del dataset.
        basename: nome base del file senza estensione.
        extension: estensione del file (es. '.jpg', '.png').

    Returns:
        Path completo del file.
    """
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    """
    Costruisce un path completo per Cityscapes mantenendo il nome relativo.

    Args:
        root: directory root del dataset.
        name: percorso relativo del file.

    Returns:
        Path completo del file.
    """
    return os.path.join(root, f'{name}')

def image_basename(filename):
    """
    Estrae il nome base di un file rimuovendo estensione e directory.

    Args:
        filename: path completo o nome file.

    Returns:
        Nome base del file senza estensione.
    """
    return os.path.basename(os.path.splitext(filename)[0])


class VOC12(Dataset):
    """
    Dataset loader per Pascal VOC 2012.

    Carica immagini e relative label semantic segmentation da cartelle
    separate (`images` e `labels`), applicando trasformazioni opzionali.
    """

    def __init__(self, root, input_transform=None, target_transform=None):
        """
        Inizializza il dataset VOC12.

        Args:
            root: directory root contenente le cartelle 'images' e 'labels'.
            input_transform: trasformazione opzionale per le immagini.
            target_transform: trasformazione opzionale per le label.
        """
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Carica una coppia immagine-label dal dataset VOC12.

        Args:
            index: indice del campione.

        Returns:
            Tuple (image, label) eventualmente trasformate.
        """
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        """
        Restituisce il numero totale di campioni nel dataset VOC12.

        Returns:
            Numero di immagini disponibili.
        """
        return len(self.filenames)


class cityscapes(Dataset):
    """
    Dataset loader per Cityscapes.

    Carica immagini RGB e label semantic segmentation (trainIds)
    dai subset del dataset Cityscapes (train/val/test).
    """

    def __init__(self, root, input_transform=None, target_transform=None, subset='val'):
        """
        Inizializza il dataset Cityscapes.

        Args:
            root: directory root del dataset Cityscapes.
            input_transform: trasformazione per le immagini.
            target_transform: trasformazione per le label.
            subset: split del dataset ('train', 'val', 'test').
        """

        self.images_root = os.path.join(root, 'leftImg8bit/' + subset)
        self.labels_root = os.path.join(root, 'gtFine/' + subset)
        print(self.images_root, self.labels_root)

        self.filenames = [
            os.path.join(dp, f)
            for dp, dn, fn in os.walk(os.path.expanduser(self.images_root))
            for f in fn if is_image(f)
        ]
        self.filenames.sort()

        self.filenamesGt = [
            os.path.join(dp, f)
            for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root))
            for f in fn if is_label(f)
        ]
        self.filenamesGt.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Carica una immagine Cityscapes e la relativa label.

        Args:
            index: indice del campione.

        Returns:
            Tuple (image, label, filename, filenameGt).
        """
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, filename, filenameGt

    def __len__(self):
        """
        Restituisce il numero di campioni Cityscapes disponibili.

        Returns:
            Numero totale di immagini nel dataset.
        """
        return len(self.filenames)