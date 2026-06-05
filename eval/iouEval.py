# Code for evaluating IoU 
# Nov 2017
# Eduardo Romera
#######################

import torch

class iouEval:
    """
    Valutatore per il calcolo della Intersection over Union (IoU).

    La classe accumula, batch dopo batch, i conteggi di veri positivi, falsi
    positivi e falsi negativi per ogni classe semantica. A partire da questi
    conteggi calcola poi la IoU media e la IoU per singola classe.

    E' pensata per valutare modelli di semantic segmentation, come ERFNet, su
    predizioni pixel-wise. Supporta sia tensori gia' in formato one-hot
    (`batch_size x nClasses x H x W`) sia mappe di etichette con un solo canale
    (`batch_size x 1 x H x W`), che vengono convertite internamente in one-hot.
    """

    def __init__(self, nClasses, ignoreIndex=19):
        """
        Inizializza il valutatore IoU.

        `nClasses` indica il numero totale di classi considerate. 
        `ignoreIndex` indica l'indice della classe da ignorare durante la valutazione, 
        ad esempio una classe void/unlabeled. Se l'indice da ignorare e' maggiore
        del numero di classi disponibili, non viene ignorata nessuna classe.
        """
        self.nClasses = nClasses
        self.ignoreIndex = ignoreIndex if nClasses>ignoreIndex else -1 #if ignoreIndex is larger than nClasses, consider no ignoreIndex
        self.reset()

    def reset (self):
        """
        Azzera i contatori interni usati per calcolare la IoU.

        Crea tre vettori, uno per classe valutata, che accumulano veri
        positivi (`tp`), falsi positivi (`fp`) e falsi negativi (`fn`). Se e'
        presente una classe da ignorare, questa viene esclusa dai contatori.
        """
        classes = self.nClasses if self.ignoreIndex==-1 else self.nClasses-1
        self.tp = torch.zeros(classes).double()
        self.fp = torch.zeros(classes).double()
        self.fn = torch.zeros(classes).double()        

    def addBatch(self, x, y):   #x=preds, y=targets
        """
        Aggiunge un batch di predizioni e ground truth alla valutazione.

        `x` contiene le predizioni del modello,
        `y` contiene le etichette ground truth. 
        Entrambi possono essere mappe di classi a un canale oppure tensori one-hot. 
        La funzione converte i tensori quando necessario, gestisce l'eventuale classe da ignorare e
        aggiorna i conteggi cumulativi di veri positivi, falsi positivi e falsi negativi per ogni classe.
        """
        #sizes should be "batch_size x nClasses x H x W"
                
        #print ("X is cuda: ", x.is_cuda)
        #print ("Y is cuda: ", y.is_cuda)

        if (x.is_cuda or y.is_cuda):
            x = x.cuda()
            y = y.cuda()

        #if size is "batch_size x 1 x H x W" scatter to onehot
        if (x.size(1) == 1):
            x_onehot = torch.zeros(x.size(0), self.nClasses, x.size(2), x.size(3))  
            if x.is_cuda:
                x_onehot = x_onehot.cuda()
            x_onehot.scatter_(1, x, 1).float()
        else:
            x_onehot = x.float()

        if (y.size(1) == 1):
            y_onehot = torch.zeros(y.size(0), self.nClasses, y.size(2), y.size(3))
            if y.is_cuda:
                y_onehot = y_onehot.cuda()
            y_onehot.scatter_(1, y, 1).float()
        else:
            y_onehot = y.float()

        if (self.ignoreIndex != -1): 
            ignores = y_onehot[:,self.ignoreIndex].unsqueeze(1)
            x_onehot = x_onehot[:, :self.ignoreIndex]
            y_onehot = y_onehot[:, :self.ignoreIndex]
        else:
            ignores=0

        #print(type(x_onehot))
        #print(type(y_onehot))
        #print(x_onehot.size())
        #print(y_onehot.size())

        tpmult = x_onehot * y_onehot    #times prediction and gt coincide is 1
        tp = torch.sum(torch.sum(torch.sum(tpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fpmult = x_onehot * (1-y_onehot-ignores) #times prediction says its that class and gt says its not (subtracting cases when its ignore label!)
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fnmult = (1-x_onehot) * (y_onehot) #times prediction says its not that class and gt says it is
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze() 

        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()

    def getIoU(self):
        """
        Calcola la IoU media e la IoU per classe.

        Per ogni classe usa la formula `IoU = TP / (TP + FP + FN)`. 
        Il valore `1e-15` nel denominatore evita divisioni per zero. 
        Restituisce una tupla composta da IoU media e vettore delle IoU per classe.
        """
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        return torch.mean(iou), iou     #returns "iou mean", "iou per class"

# Class for colors
class colors:
    """
    Contenitore di codici ANSI per colorare l'output nel terminale.

    Gli attributi della classe rappresentano sequenze di escape usate per
    stampare testo colorato o formattato, ad esempio rosso, verde, grassetto o
    sottolineato. `ENDC` serve a ripristinare il colore/formato predefinito.
    """
    RED       = '\033[31;1m'
    GREEN     = '\033[32;1m'
    YELLOW    = '\033[33;1m'
    BLUE      = '\033[34;1m'
    MAGENTA   = '\033[35;1m'
    CYAN      = '\033[36;1m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC      = '\033[0m'

# Colored value output if colorized flag is activated.
def getColorEntry(val):
    """
    Restituisce il colore ANSI associato a un valore numerico di prestazione.

    La funzione viene usata per rendere piu' leggibile l'output delle metriche:
    valori bassi vengono colorati in rosso o giallo, valori intermedi in blu o
    ciano e valori alti in verde. Se `val` non e' un numero float, viene
    restituito direttamente il codice di reset `colors.ENDC`.
    """
    if not isinstance(val, float):
        return colors.ENDC
    if (val < .20):
        return colors.RED
    elif (val < .40):
        return colors.YELLOW
    elif (val < .60):
        return colors.BLUE
    elif (val < .80):
        return colors.CYAN
    else:
        return colors.GREEN

