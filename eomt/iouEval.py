# Code for evaluating IoU 
# Nov 2017
# Eduardo Romera
#######################

import torch

class iouEval:
    """
    Accumulatore per il calcolo della Intersection over Union semantica.

    La classe mantiene, per ogni classe, il conteggio di veri positivi, falsi
    positivi e falsi negativi su tutti i batch valutati. Alla fine restituisce
    sia la mean IoU sia la IoU separata per classe.
    """

    def __init__(self, nClasses, ignoreIndex=19):
        """
        Inizializza l'accumulatore IoU.

        Args:
            nClasses: Numero totale di classi considerate, inclusa
                eventualmente la classe di ignore.
            ignoreIndex: Indice della classe da ignorare nel calcolo della IoU.
                Se l'indice non rientra nel numero di classi, non viene
                applicato nessun ignore.
        """
        self.nClasses = nClasses
        self.ignoreIndex = ignoreIndex if nClasses>ignoreIndex else -1 #if ignoreIndex is larger than nClasses, consider no ignoreIndex
        self.reset()

    def reset (self):
        """
        Azzera i conteggi accumulati di veri positivi, falsi positivi e falsi
        negativi.

        Returns:
            None.
        """
        classes = self.nClasses if self.ignoreIndex==-1 else self.nClasses-1
        self.tp = torch.zeros(classes).double()
        self.fp = torch.zeros(classes).double()
        self.fn = torch.zeros(classes).double()        

    def addBatch(self, x, y):   #x=preds, y=targets
        """
        Aggiorna i conteggi IoU usando un batch di predizioni e label.

        Le predizioni e le label possono essere gia in formato one-hot oppure
        in formato `[batch, 1, H, W]` con indici di classe. Il metodo converte
        quando necessario, rimuove la classe da ignorare e somma TP, FP e FN
        alle statistiche interne.

        Args:
            x: Predizioni del modello, come indici di classe o tensore one-hot.
            y: Maschere ground truth, come indici di classe o tensore one-hot.

        Returns:
            None.
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
        Calcola la IoU finale a partire dai conteggi accumulati.

        Returns:
            Tupla `(mean_iou, iou_per_class)` con la media sulle classi valide
            e il vettore delle IoU classe per classe.
        """
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        return torch.mean(iou), iou     #returns "iou mean", "iou per class"

# Class for colors
class colors:
    """
    Codici ANSI usati per colorare l'output testuale delle metriche.

    I colori vengono usati solo in console per rendere piu leggibili i valori
    di IoU in base alla qualita del risultato.
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
    Restituisce un colore ANSI in base al valore della IoU.

    Valori bassi vengono mostrati in rosso o giallo, valori intermedi in blu o
    ciano e valori alti in verde. Se il valore non e un `float`, viene
    restituito il codice di reset.

    Args:
        val: Valore IoU da rappresentare.

    Returns:
        Stringa con il codice colore ANSI corrispondente.
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

