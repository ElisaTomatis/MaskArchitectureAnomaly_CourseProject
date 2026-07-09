# Mask Architecture Anomaly Segmentation for Road Scenes

This repository contains the code developed for the **Mask Architecture Anomaly Segmentation for Road Scenes** course project. The goal is to study anomaly segmentation in autonomous driving scenarios, compare pixel-based and mask-based baselines, and extend EoMT with an Outlier Exposure fine-tuning strategy.

The main contribution of this project is a **Fine-tune with Outlier Exposure** pipeline for EoMT. During Cityscapes training, objects are extracted from COCO instance annotations, pasted into Cityscapes images as synthetic out-of-distribution regions, and used to optimize the model with an anomaly-aware objective. This makes the model explicitly observe unknown-like objects during fine-tuning instead of relying only on post-hoc anomaly scores at inference time.

## Project Overview

The project follows three main stages:

1. **ERFNet pixel-based baselines**
   - Evaluation of a pretrained ERFNet model on road anomaly datasets.
   - Post-hoc anomaly scores: MSP, Max Logit, and Max Entropy.

2. **EoMT mask-based baselines**
   - Adaptation of the evaluation code to the EoMT mask architecture.
   - Post-hoc anomaly scores: MSP, Max Logit, Max Entropy, and RbA.
   - Temperature scaling experiments for calibrated MSP-based anomaly detection.

3. **Outlier Exposure fine-tuning**
   - Fine-tuning EoMT on Cityscapes enriched with pasted COCO objects.
   - Synthetic OOD masks are generated at training time.
   - A Rejected-by-All inspired loss is added on pasted OOD pixels.
   - The default setup freezes the backbone and updates only the prediction heads, which is more practical under limited computational resources.

## Repository Structure

```text
.
+-- eomt/
|   +-- configs/dinov2/cityscapes/semantic/
|   |   +-- eomt_base_640.yaml        # Standard Cityscapes EoMT configuration
|   |   +-- oe_config.yaml            # Outlier Exposure fine-tuning configuration
|   +-- datasets/
|   |   +-- cityscapes_semantic.py    # Standard Cityscapes data module
|   |   +-- cityscapes_semantic_oe.py # Cityscapes + COCO cut-paste OOD wrapper
|   +-- training/
|   |   +-- mask_classification_semantic.py
|   |   +-- mask_classification_semantic_oe.py
|   +-- evalAnomalyEomt.py
|   +-- evalAnomalyEoMTTemperatureOptmized.py
|   +-- eval_iouEoMT.py
|   +-- main.py
+-- eval/
|   +-- evalAnomalyERFNet.py
|   +-- evalAnomalyERFNetTemperatureOptimazed.py
|   +-- eval_iouERFNet.py
+-- trained_models/
|   +-- erfnet_pretrained.pth
|   +-- erfnet_encoder_pretrained.pth.tar
+-- README.md
```

## Outlier Exposure Extension

The Outlier Exposure extension is implemented in:

- `eomt/datasets/cityscapes_semantic_oe.py`
- `eomt/training/mask_classification_semantic_oe.py`
- `eomt/configs/dinov2/cityscapes/semantic/oe_config.yaml`

### Synthetic OOD Generation

`CityscapesSemanticOE` extends the standard Cityscapes data module. During training, it wraps the Cityscapes training set with `OODDatasetWrapper`, which probabilistically applies a COCO cut-paste transformation.

The pipeline is:

1. Select a COCO object from categories such as `elephant`, `giraffe`, `zebra`, `bear`, `chair`, `couch`, `microwave`, `banana`, `apple`, or `backpack`.
2. Convert the COCO instance annotation into a binary object mask.
3. Crop the object and its mask from the COCO image.
4. Resize the object while preserving its aspect ratio.
5. Paste it at a random location on a Cityscapes image.
6. Create an `ood_mask` marking exactly the pasted pixels.
7. Remove pasted pixels from the standard semantic supervision masks so that they are treated as out-of-distribution regions.

The relevant configuration parameters are defined in `oe_config.yaml`:

```yaml
data:
  class_path: datasets.cityscapes_semantic_oe.CityscapesSemanticOE
  init_args:
    coco_root: null
    p_ood: 0.15
    coco_split: val2017
    ood_target_height_range: [80, 250]
```

If `coco_root` is left as `null`, the code expects the environment variable `EOMT_COCO_ROOT` to point to the COCO root directory.

### RbA-Based Fine-Tuning Objective

The class `MaskClassificationSemanticOE` extends the standard EoMT semantic segmentation Lightning module. It keeps the original EoMT segmentation losses and adds an auxiliary RbA hinge loss on pasted OOD pixels.

For each pasted OOD pixel, the model is encouraged to reject all known Cityscapes classes. The loss is controlled by:

```yaml
model:
  init_args:
    lambda_rba: 0.1
    rba_alpha: 5.0
    rba_reduction: mean
    freeze_heads_only: True
```

The final training objective is:

```text
total_loss = standard_eomt_segmentation_loss + lambda_rba * rba_hinge_loss
```

By default, `freeze_heads_only: True` freezes the EoMT backbone and only trains `class_head` and `mask_head`. This choice follows the project constraints: it reduces memory usage and training time while still allowing the prediction heads to adapt to OOD-aware supervision.

## Installation

The EoMT code uses Python and PyTorch Lightning. A Conda environment is recommended.

```bash
conda create -n eomt python=3.13.2
conda activate eomt
cd eomt
python -m pip install -r requirements.txt
```

Weights & Biases is used for experiment logging:

```bash
wandb login
```

## Dataset Preparation

### Cityscapes

Download the Cityscapes files required by the EoMT data module and place them in a dataset directory. The original EoMT data module can read the zipped Cityscapes files directly, so they do not need to be extracted.

Typical files include:

- `leftImg8bit_trainvaltest.zip`
- `gtFine_trainvaltest.zip`

Pass the directory containing these files through `--data.path`.

### COCO for Outlier Exposure

For the Outlier Exposure extension, prepare COCO with images and instance annotations. The expected structure is:

```text
/path/to/coco/
+-- val2017/
|   +-- *.jpg
+-- annotations/
    +-- instances_val2017.json
```

The COCO path can be provided either in the command line:

```bash
--data.init_args.coco_root /path/to/coco
```

or through an environment variable:

```bash
export EOMT_COCO_ROOT=/path/to/coco
```

On Windows PowerShell:

```powershell
$env:EOMT_COCO_ROOT = "C:\path\to\coco"
```

### Anomaly Segmentation Evaluation Datasets

The evaluation scripts expect anomaly datasets with an `images` folder and a matching `labels_masks` folder. The helper functions map image paths to their masks by replacing `images` with `labels_masks`.

Supported dataset naming conventions include:

- RoadAnomaly / RoadAnomaly21
- RoadObsticle21
- Fishyscapes Lost & Found
- Fishyscapes Static
- StreetHazard

## Running the Main Experiments

Run commands from the corresponding source directory unless stated otherwise.

### 1. ERFNet Anomaly Baseline

```bash
cd eval
python evalAnomalyERFNet.py \
  --input "/path/to/RoadAnomaly21/images/*.jpg" \
  --loadDir "../trained_models/" \
  --loadWeights "erfnet_pretrained.pth"
```

This evaluates ERFNet with Max Logit, MSP, and Max Entropy scores. Results are printed to the console and appended to `eval/results.txt`.

### 2. ERFNet Temperature Scaling

```bash
cd eval
python evalAnomalyERFNetTemperatureOptimazed.py \
  --input "/path/to/RoadAnomaly21/images/*.jpg" \
  --loadDir "../trained_models/" \
  --loadWeights "erfnet_pretrained.pth"
```

The optimized version stores logits once and evaluates several temperature values without repeating the model forward pass.

### 3. EoMT Anomaly Baseline

```bash
cd eomt
python evalAnomalyEomt.py \
  --input "/path/to/RoadAnomaly21/images/*.jpg" \
  --weights_dir "/path/to/eomt_cityscapes.bin"
```

This evaluates EoMT using Max Logit, MSP, Max Entropy, and RbA anomaly scores.

### 4. EoMT Temperature Scaling

```bash
cd eomt
python evalAnomalyEoMTTemperatureOptmized.py \
  --input "/path/to/RoadAnomaly21/images/*.jpg"
```

Before running this script, check the `state_dict_path` variable inside the file and set it to the EoMT checkpoint to evaluate.

### 5. Cityscapes mIoU Evaluation for EoMT

```bash
cd eomt
python eval_iouEoMT.py \
  --datadir "/path/to/cityscapes" \
  --subset val \
  --batch-size 1
```

This reports the per-class IoU and mean IoU on Cityscapes.

### 6. Fine-Tuning EoMT with Outlier Exposure

```bash
cd eomt
python main.py fit \
  -c configs/dinov2/cityscapes/semantic/oe_config.yaml \
  --data.path /path/to/cityscapes \
  --data.init_args.coco_root /path/to/coco \
  --model.ckpt_path /path/to/eomt_cityscapes.bin \
  --model.load_ckpt_class_head False \
  --trainer.devices 1 \
  --data.batch_size 1
```

Important options:

- `--data.path`: directory containing the Cityscapes data.
- `--data.init_args.coco_root`: COCO root used for pasted OOD objects.
- `--model.ckpt_path`: pretrained EoMT checkpoint used as the fine-tuning starting point.
- `--model.load_ckpt_class_head False`: useful when loading checkpoints with a different head setup.
- `--trainer.devices`: number of GPUs.
- `--data.batch_size`: batch size per device.

The default `oe_config.yaml` runs for 10 epochs and logs the experiment under the W&B name `cityscapes_semantic_eomt_base_640_oe`.

Checkpoint names include the learning rate and OE loss weight, making it easier to compare fine-tuning runs:

```text
lr<lr>-lambdaoe<lambda>-epoch<epoch>-step<step>.ckpt
```

## Evaluation Metrics

The anomaly segmentation scripts report:

- **AuPRC**: Average Precision for detecting OOD pixels.
- **FPR@TPR95**: false positive rate when true positive rate is fixed at 95%.
- **mIoU**: semantic segmentation quality on Cityscapes.

In general, better anomaly segmentation corresponds to higher AuPRC and lower FPR@TPR95.

## Notes on Paths and Checkpoints

Some scripts were originally used in a Colab/Drive environment and may contain default checkpoint paths such as `/content/drive/...`. When running locally, replace those paths with the location of the desired checkpoint.

The ERFNet pretrained weights are included in `trained_models/`. EoMT checkpoints are not stored directly in this repository and should be downloaded or generated separately.

## References

This project builds on the literature and tools indicated in the course assignment, including:

- ERFNet for real-time semantic segmentation.
- MaskFormer and Mask2Former for mask-based segmentation.
- DINOv2 and EoMT for transformer-based segmentation.
- SegmentMeIfYouCan and Fishyscapes anomaly segmentation benchmarks.
- RbA: Segmenting Unknown Regions Rejected by All.
- Temperature scaling for confidence calibration.
- Cityscapes and COCO datasets.
