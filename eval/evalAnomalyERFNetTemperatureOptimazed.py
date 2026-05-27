# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
import shutil
import numpy as np
from PIL import Image
from erfnet import ERFNet
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor
from functions import *

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
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'w')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print("Model and weights LOADED successfully")
    model.eval()

    # ---------------------------------------------------------------
    # FASE 1: inferenza — salva i logit su disco, tieni solo le GT
    # ---------------------------------------------------------------
    os.makedirs('temp_logits', exist_ok=True)
    ood_gts_list = []
    valid_paths = []

    for path in sorted(glob.glob(os.path.expanduser(str(args.input[0])))):
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float()
        if not args.cpu:
            images = images.cuda()

        with torch.no_grad():
            result = model(images)
        result = result.squeeze(0)  # shape: [NUM_CLASSES, H, W]

        # controlla la GT prima di salvare i logit
        pathGT = create_pathGT(path)
        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = create_oodgts(mask, pathGT).astype(np.uint8)

        if 1 not in np.unique(ood_gts):
            print(f"Saltato {path}: nessuna anomalia trovata.")
            del result, mask, ood_gts
            if not args.cpu:
                torch.cuda.empty_cache()
            continue

        # salva i logit su disco solo se l'immagine è valida
        filename = os.path.basename(path)
        np.save(f'temp_logits/{filename}.npy', result.data.cpu().numpy())

        ood_gts_list.append(ood_gts)
        valid_paths.append(path)

        del result, mask, ood_gts
        if not args.cpu:
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # FASE 2: valutazione — itera per temperatura, rileggi dal disco
    # ---------------------------------------------------------------
    file.write("\n")

    t_vec = np.concatenate((np.array((0.5, 0.75, 1.1)), np.exp(np.linspace(np.log(0.1), np.log(50), 20))))
    auprc_list = []
    fpr_list = []

    for i, t in enumerate(t_vec):
        current_t_scores = []

        for path in valid_paths:
            filename = os.path.basename(path)
            logits = np.load(f'temp_logits/{filename}.npy')  # [NUM_CLASSES, H, W]

            logits_t = torch.from_numpy(logits) / t
            probs_tensor = torch.nn.functional.softmax(logits_t, dim=0)
            anomaly_result_softmax = (1.0 - torch.max(probs_tensor, dim=0)[0]).numpy()
            current_t_scores.append(anomaly_result_softmax)

            del logits, logits_t, probs_tensor

        [prc_auc_softmax, fpr_softmax] = eval_score(ood_gts_list, current_t_scores)
        auprc_list.append(prc_auc_softmax)
        fpr_list.append(fpr_softmax)

        if i <= 2:
            file.write(f'\n  t = {t} -->  ' + 'AUPRC softmax score:' + str(prc_auc_softmax * 100.0) + '   FPR@TPR95 softmax:' + str(fpr_softmax * 100.0))
            print(f't = {t}')
            print(f'AUPRC softmax score: {prc_auc_softmax * 100.0}')
            print(f'FPR@TPR95 softmax: {fpr_softmax * 100.0}')

        del current_t_scores

    performance_array = np.array(auprc_list) - np.array(fpr_list) + 1.0
    best_index = np.argmax(performance_array)
    best_t = t_vec[best_index]
    best_auprc = auprc_list[best_index]
    best_fpr = fpr_list[best_index]
    file.write(f'\n best t = {best_t} -->  ' + 'AUPRC softmax score:' + str(best_auprc * 100.0) + '   FPR@TPR95 softmax:' + str(best_fpr * 100.0))

    file.close()

    # pulizia cartella temporanea
    shutil.rmtree('temp_logits')

if __name__ == '__main__':
    main()

