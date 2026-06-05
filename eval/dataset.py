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
    Carica un'immagine da file usando PIL

    Riceve un file path o un file object aperto e restituisce l'oggetto `Image` corrispondente
    La conversione del formato viene gestita nelle classi dataset, 
    dove le immagini sono trasformate in RGB e le label in palette mode
    """
    return Image.open(file)

def is_image(filename):
    """
    Verifica se un file ha un'estensione immagine supportata

    Restituisce `True` per file con estensione `.jpg` o `.png`, cioe' i formati
    previsti dai loader VOC12 e Cityscapes presenti in questo progetto
    """
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    """
    Verifica se un file e' una label Cityscapes in formato trainId

    Le ground truth Cityscapes usate per il training/evaluation terminano con
    `_labelTrainIds.png`, questa funzione filtra quei file
    """
    return filename.endswith("_labelTrainIds.png")

def image_path(root, basename, extension):
    """
    Costruisce il percorso di un file a partire da cartella, nome base ed estensione

    Viene usata dal dataset VOC12 per ricostruire il percorso dell'immagine o
    della maschera partendo dal nome senza estensione
    """
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    """
    Costruisce il percorso di un file Cityscapes

    Riceve la root del dataset e il nome/percorso relativo del file, poi li combina con `os.path.join`
    Nel loader Cityscapes serve per aprire immagini e label trovate durante la scansione delle sottocartelle
    """
    return os.path.join(root, f'{name}')

def image_basename(filename):
    """
    Estrae il nome base di un file rimuovendo cartella ed estensione

    E' utile per accoppiare immagini e label che condividono lo stesso nome ma
    si trovano in cartelle diverse o hanno estensioni diverse
    """
    return os.path.basename(os.path.splitext(filename)[0])

class VOC12(Dataset):
    """
    Dataset PyTorch per immagini e label in stile VOC12

    La classe legge immagini dalla sottocartella `images` e maschere dalla
    sottocartella `labels`, accoppiandole tramite il nome base del file
    Restituisce coppie `(image, label)` eventualmente trasformate con le
    trasformazioni passate al costruttore
    """
    def __init__(self, root, input_transform=None, target_transform=None):
        """
        Inizializza il dataset VOC12

        `root` e' la directory principale del dataset
        `input_transform` viene applicata alle immagini RGB
        `target_transform` viene applicata alle maschere di label
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
        Restituisce immagine e label all'indice richiesto

        L'immagine viene letta come RGB, la label come immagine con palette, 
        poi entrambe vengono trasformate se sono state definite trasformazioni
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
        Restituisce il numero totale di esempi nel dataset
        """
        return len(self.filenames)


class cityscapes(Dataset):
    """
    Dataset PyTorch per Cityscapes

    La classe cerca ricorsivamente le immagini in `leftImg8bit/<subset>` e le
    label in `gtFine/<subset>`, usando solo le ground truth in formato `_labelTrainIds.png`
    Restituisce immagine, label e i rispettivi percorsi
    informazione utile durante la valutazione e la stampa dei risultati
    """
    def __init__(self, root, input_transform=None, target_transform=None, subset='val'):
        """
        Inizializza il dataset Cityscapes

        `root` e' la directory principale di Cityscapes, 
        `subset` indica la partizione da usare, ad esempio `val` o `train`
        Le trasformazioni opzionali vengono applicate separatamente a immagini e label
        """
        self.images_root = os.path.join(root, 'leftImg8bit/' + subset)
        self.labels_root = os.path.join(root, 'gtFine/' + subset)
        print(self.images_root, self.labels_root)
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Restituisce un campione Cityscapes

        Carica immagine RGB e label corrispondente, applica le trasformazioni
        definite e restituisce `(image, label, filename, filenameGt)`
        """
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        #print(filename)

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
        Restituisce il numero di immagini disponibili nel subset selezionato
        """
        return len(self.filenames)

