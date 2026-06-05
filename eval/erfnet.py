# ERFNET full network definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class DownsamplerBlock (nn.Module):
    """
    Blocco di downsampling usato nell'encoder di ERFNet

    Riduce la risoluzione spaziale della feature map di un fattore 2 e aumenta il numero di canali. 
    Il blocco combina due rami: una convoluzione 3x3 con stride 2 e un max pooling. 
    I risultati vengono concatenati sui canali, normalizzati con BatchNorm e passati attraverso una ReLU.
    """

    def __init__(self, ninput, noutput):
        """
        Inizializza il blocco di downsampling.

        `ninput` e' il numero di canali in ingresso, mentre `noutput` e' il
        numero di canali in uscita dopo la concatenazione tra convoluzione e max pooling.
        """
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        """
        Esegue il downsampling della feature map in ingresso.

        Applica in parallelo convoluzione e max pooling, concatena i due output
        lungo la dimensione dei canali e restituisce il risultato normalizzato e attivato con ReLU.
        """
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)
    

class non_bottleneck_1d (nn.Module):
    """
    Blocco residuale non-bottleneck 1D di ERFNet.

    Implementa convoluzioni 3x1 e 1x3 al posto di convoluzioni 3x3 complete, riducendo il costo computazionale. 
    La seconda coppia di convoluzioni puo' usare dilatazione per aumentare il campo ricettivo senza ridurre 
    la risoluzione. Il risultato viene sommato all'input tramite connessione residuale.
    """

    def __init__(self, chann, dropprob, dilated):        
        """
        Inizializza il blocco residuale fattorizzato.

        `chann` indica il numero di canali in ingresso e uscita, 
        `dropprob` la probabilita' di dropout spaziale, 
        `dilated` controlla la dilatazione della seconda coppia di convoluzioni.
        """
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):
        """
        Applica il blocco residuale alla feature map in ingresso.

        Esegue due coppie di convoluzioni fattorizzate 3x1 e 1x3, con normalizzazione e ReLU intermedie. 
        Se configurato, applica dropout e poi somma l'output all'input originale prima della ReLU finale.
        """

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)


class Encoder(nn.Module):
    """
    Encoder di ERFNet per l'estrazione di feature semantiche.

    L'encoder trasforma un'immagine RGB in una rappresentazione compatta a
    bassa risoluzione usando blocchi di downsampling e blocchi residuali `non_bottleneck_1d`. 
    In modalita' `predict` puo' anche produrre direttamente logits di classe a bassa risoluzione 
    tramite una convoluzione 1x1.
    """

    def __init__(self, num_classes):
        """
        Costruisce l'encoder ERFNet.

        `num_classes` indica il numero di classi semantiche da predire quando
        l'encoder viene usato da solo in modalita' `predict=True`.
        """
        super().__init__()
        self.initial_block = DownsamplerBlock(3,16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16,64))

        for x in range(0, 5):    #5 times
            self.layers.append(non_bottleneck_1d(64, 0.1, 1))  

        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 2):    #2 times
            self.layers.append(non_bottleneck_1d(128, 0.1, 2))
            self.layers.append(non_bottleneck_1d(128, 0.1, 4))
            self.layers.append(non_bottleneck_1d(128, 0.1, 8))
            self.layers.append(non_bottleneck_1d(128, 0.1, 16))

        #only for encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        """
        Esegue il forward pass dell'encoder.

        Se `predict` e' `False`, restituisce la feature map codificata a 128 canali. 
        Se `predict` e' `True`, applica anche la convoluzione finale e
        restituisce logits di classe a bassa risoluzione.
        """
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class UpsamplerBlock (nn.Module):
    """
    Blocco di upsampling usato nel decoder di ERFNet.

    Aumenta la risoluzione spaziale della feature map tramite convoluzione
    trasposta, poi applica BatchNorm e ReLU. Serve a ricostruire gradualmente
    una mappa di segmentazione alla risoluzione dell'immagine.
    """

    def __init__(self, ninput, noutput):
        """
        Inizializza il blocco di upsampling.

        `ninput` e' il numero di canali della feature map in ingresso, 
        `noutput` e' il numero di canali prodotti dalla convoluzione trasposta.
        """
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        """
        Esegue l'upsampling della feature map.

        Applica una convoluzione trasposta per raddoppiare la risoluzione,
        normalizza il risultato e restituisce l'attivazione ReLU.
        """
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class Decoder (nn.Module):
    """
    Decoder di ERFNet per produrre la segmentazione finale.

    Riceve le feature compatte dell'encoder e le riporta progressivamente a una
    risoluzione piu' alta tramite blocchi di upsampling e blocchi residuali.
    L'ultimo layer produce i logits finali per ogni classe semantica.
    """

    def __init__(self, num_classes):
        """
        Costruisce il decoder ERFNet.

        `num_classes` definisce il numero di canali dell'output finale, cioe'
        una mappa di logits per ciascuna classe di segmentazione.
        """
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128,64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64,16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d( 16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        """
        Esegue il forward pass del decoder.

        Applica in sequenza i blocchi di upsampling e raffinamento, poi usa la convoluzione 
        trasposta finale per ottenere la mappa di logits alla risoluzione di uscita.
        """
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


class ERFNet(nn.Module):
    """
    Modello ERFNet completo per semantic segmentation.

    Combina un encoder e un decoder per trasformare un'immagine RGB in una mappa di logits pixel-wise. 
    Nel progetto viene usato come baseline pixel-based per valutare metodi post-hoc di anomaly segmentation, 
    come MSP, Max Logit, Max Entropy e varianti con temperature scaling.
    """

    def __init__(self, num_classes, encoder=None):  #use encoder to pass pretrained encoder
        """
        Inizializza il modello ERFNet.

        `num_classes` e' il numero di classi semantiche in output. 
        Il parametro opzionale `encoder` permette di passare un encoder gia' costruito o
        pre-addestrato; se non viene fornito, viene creato un nuovo `Encoder`.
        """
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

    def forward(self, input, only_encode=False):
        """
        Esegue il forward pass del modello.

        Se `only_encode` e' `True`, usa solo l'encoder in modalita' predittiva e
        restituisce logits a bassa risoluzione. Altrimenti passa l'immagine
        attraverso encoder e decoder, restituendo la mappa di logits finale per
        la segmentazione semantica.
        """
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)    #predict=False by default
            return self.decoder.forward(output)

