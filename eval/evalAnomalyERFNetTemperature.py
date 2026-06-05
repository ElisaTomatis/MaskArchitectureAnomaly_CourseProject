# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor
from functions import *

"""
Script di valutazione anomaly segmentation con temperature scaling per ERFNet.

Il file carica un modello ERFNet pre-addestrato e valuta lo score MSP
(`1 - max(softmax)`) applicando diverse temperature ai logits prima della
softmax. La temperature scaling modifica la confidenza della distribuzione di
probabilita' e puo' migliorare la capacita' del modello di separare pixel
in-distribution e pixel anomali/OoD.

Per ogni temperatura vengono calcolate le metriche AuPRC e FPR95; lo script
riporta anche la temperatura migliore secondo una misura combinata basata su
AuPRC alto e FPR basso.
"""

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


def main():
    """
    Esegue la ricerca della temperatura per MSP su ERFNet.

    La funzione legge gli argomenti da riga di comando, carica ERFNet con i
    pesi specificati e prepara un vettore di temperature da testare. Per ogni
    immagine indicata da `--input`, esegue il forward pass del modello una sola
    volta e poi ricalcola la softmax dei logits per ciascun valore di temperatura. 
    Da ogni softmax ricava lo score di anomalia MSP, definito come `1 - probabilita' massima`.

    Le ground truth vengono recuperate con `create_pathGT`, ridimensionate e
    convertite in maschere binarie OoD tramite `create_oodgts`. 
    Le immagini che non contengono pixel anomali vengono ignorate. 
    Al termine, per ogni temperatura, `eval_score` calcola AuPRC e FPR@TPR95
    i risultati principali e la temperatura migliore vengono salvati in `results.txt`.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    anomaly_score_softmax_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'w')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")
    model.eval()
    
    t_vec = np.concatenate((np.array((0.5,0.75,1.1)), np.exp(np.linspace(np.log(0.1), np.log(50), 20))))    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()
        with torch.no_grad():
            result = model(images)
        result = result.squeeze(0)
        
        anomaly_result_list = []
        for t in t_vec:
            probs_tensor = torch.nn.functional.softmax(result.data.cpu() / t, dim=0)  
            anomaly_result_softmax = 1.0 - np.max(probs_tensor.numpy(), axis=0)
            anomaly_result_list.append(anomaly_result_softmax)

        pathGT = create_pathGT(path)  

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = create_oodgts(mask, pathGT)
        
        if 1 not in np.unique(ood_gts):
            continue              
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_softmax_list.append(anomaly_result_list)
        del result, ood_gts, anomaly_result_list, mask
        torch.cuda.empty_cache()

    file.write( "\n")

    scores_array = np.array(anomaly_score_softmax_list)
    auprc_list = []
    fpr_list = []
    for i,t in enumerate(t_vec):
        [prc_auc_softmax, fpr_softmax] = eval_score(ood_gts_list, scores_array[:, i])
        auprc_list.append(prc_auc_softmax)
        fpr_list.append(fpr_softmax)
        if i <= 2:
            file.write(f'\n  t = {t} -->  ' + 'AUPRC softmax score:' + str(prc_auc_softmax*100.0) + '   FPR@TPR95 softmax:' + str(fpr_softmax*100.0))
            print(f't = {t}')
            print(f'AUPRC softmax score: {prc_auc_softmax*100.0}')
            print(f'FPR@TPR95 softmax: {fpr_softmax*100.0}')
    
    performance_array = np.array(auprc_list) - np.array(fpr_list) + 1.0
    best_index = np.argmax(performance_array)
    best_t = t_vec[best_index]
    best_auprc = auprc_list[best_index]
    best_fpr = fpr_list[best_index]
    file.write(f'\n best t = {best_t} -->  ' + 'AUPRC softmax score:' + str(best_auprc*100.0) + '   FPR@TPR95 softmax:' + str(best_fpr*100.0))
    
    file.close()

if __name__ == '__main__':
    main()
