# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import yaml
import glob
import torch
import random
from PIL import Image
import numpy as np
from eomt import EoMT
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
import warnings

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

input_transform = Compose([
    Resize((1280, 1280), Image.BILINEAR), # EoMT Giant usa solitamente 1280
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # Standard ImageNet/DINO
])

target_transform = Compose(
    [
        Resize((1280, 1280), Image.NEAREST),
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
    parser.add_argument('--loadDir',default="../models/")
    parser.add_argument('--loadModel', default="eomt.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    # TODO: understand if it is needed
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    model = EoMT(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    config_path = "configs/dinov2/coco/panoptic/eomt_giant_1280.yaml"  # TODO: change to the config file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    name = config.get("trainer", {}).get("logger", {}).get("init_args", {}).get("name")
    
    if name is None:
        warnings.warn("No logger name found in the config. Please specify a model name.")
    else:
        try:
            state_dict_path = hf_hub_download(
                repo_id=f"tue-mps/{name}",
                filename="pytorch_model.bin",
            )
 
            state_dict = torch.load(
                state_dict_path, weights_only=True
            )
            model.load_state_dict(state_dict, strict=False)

            print ("Model and weights LOADED successfully")

        except RepositoryNotFoundError:
            warnings.warn(
                f"Pre-trained model not found for `{name}`. Please load your own checkpoint."
            )

    model.eval()

    print('cao')

    anomaly_score_logit_list = []
    anomaly_score_softmax_list = []
    anomaly_score_entropy_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'w')

    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()
        print(images.shape)
        # images = images.permute(0,3,1,2)
        with torch.no_grad():
            result = model(images)
        result = result.squeeze(0)
        anomaly_result_logit = 1.0 - np.max(result.data.cpu().numpy(), axis=0)
        anomaly_result_softmax = 1.0 - np.max(torch.nn.functional.softmax(result.data.cpu(), dim = 0).numpy(), axis=0)
        probs_tensor = torch.nn.functional.softmax(result, dim=0)
        anomaly_result_entropy = -torch.sum(probs_tensor * torch.log(probs_tensor), dim=0).data.cpu().numpy()            
        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_logit_list.append(anomaly_result_logit)
             anomaly_score_softmax_list.append(anomaly_result_softmax)
             anomaly_score_entropy_list.append(anomaly_result_entropy)
        del result, anomaly_result_logit, ood_gts, mask
        torch.cuda.empty_cache()

    file.write( "\n")
    # TODO: functionalize all the code

    def eval_score(ood_gts_list, anomaly_score_list):
    
        ood_gts = np.array(ood_gts_list)
        anomaly_scores = np.array(anomaly_score_list)

        ood_mask = (ood_gts == 1)
        ind_mask = (ood_gts == 0)

        ood_out = anomaly_scores[ood_mask]
        ind_out = anomaly_scores[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))

        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)

        return [prc_auc, fpr]

    [prc_auc_logit, fpr_logit] = eval_score(ood_gts_list, anomaly_score_logit_list)
    [prc_auc_softmax, fpr_softmax] = eval_score(ood_gts_list, anomaly_score_softmax_list)
    [prc_auc_entropy, fpr_entropy] = eval_score(ood_gts_list, anomaly_score_entropy_list)

    print(f'AUPRC logit score: {prc_auc_logit*100.0}')
    print(f'FPR@TPR95 logit: {fpr_logit*100.0}')

    print(f'AUPRC softmax score: {prc_auc_softmax*100.0}')
    print(f'FPR@TPR95 softmax: {fpr_softmax*100.0}')

    print(f'AUPRC entropy score: {prc_auc_entropy*100.0}')
    print(f'FPR@TPR95 entropy: {fpr_entropy*100.0}')

    file.write(('    AUPRC logit score:' + str(prc_auc_logit*100.0) + '   FPR@TPR95 logit:' + str(fpr_logit*100.0) +
                '\n    AUPRC softmax score:' + str(prc_auc_softmax*100.0) + '   FPR@TPR95 softmax:' + str(fpr_softmax*100.0) +
                '\n    AUPRC entropy score:' + str(prc_auc_entropy*100.0) + '   FPR@TPR95 entropy:' + str(fpr_entropy*100.0)))
    file.close()

if __name__ == '__main__':
    main()
