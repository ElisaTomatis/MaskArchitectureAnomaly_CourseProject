import csv
import glob
import os
import warnings
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml
from lightning import seed_everything
from PIL import Image
from scipy import ndimage
from torch.nn import functional as F
from torchvision.transforms import Compose, Resize, ToTensor

from functions import compute_logits, create_oodgts, create_pathGT, eval_score, load_model


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


IGNORE_INDEX = 255


CITYSCAPES_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


CITYSCAPES_PALETTE = np.array(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ],
    dtype=np.uint8,
)


input_transform = Compose(
    [
        Resize((1024, 1024), Image.BILINEAR),
        ToTensor(),
    ]
)


target_transform = Compose(
    [
        Resize((1024, 1024), Image.NEAREST),
    ]
)


def load_eomt_for_visualization(
    config_path="configs/dinov2/cityscapes/semantic/eomt_base_640.yaml",
    state_dict_path="/content/drive/MyDrive/ml_anomaly_segmentation/eomt_cityscapes.bin",
    device=None,
):
    '''
    Carica il modello EoMT nello stesso modo usato negli script di valutazione.
    '''
    seed_everything(0, verbose=False)

    # Se device è "cpu" o CUDA non è disponibile, usa "cpu". 
    # Se viene richiesto "cuda" ma non è disponibile, la logica di fallback previene il crash.
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA richiesto ma non disponibile. Switch su CPU.")
        device = "cpu"
    elif device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )

    # Viene passato il device corretto (stringa 'cpu' o 'cuda') che verrà usato in map_location
    model = load_model(device, config, state_dict_path)
    return model, device


def load_image_like_eval_anomaly(image_path, device):
    '''
    Carica una immagine nello stesso formato usato da evalAnomalyEoMT.
    '''
    original_image = Image.open(image_path).convert("RGB")
    image_tensor = input_transform(original_image).float()
    image_tensor = (image_tensor * 255).to(torch.uint8)
    image_tensor = image_tensor.to(device)

    return original_image, image_tensor


def compute_pixel_logits_for_image(image_path, model, device):
    '''
    Esegue l'inferenza EoMT su una singola immagine e restituisce i logits pixel-wise.
    '''
    original_image, image_tensor = load_image_like_eval_anomaly(image_path, device)

    with torch.no_grad():
        pixel_logits = compute_logits([image_tensor], device, model)

    return original_image, pixel_logits


def compute_anomaly_score_maps(pixel_logits, temperatures=None):
    '''
    Calcola le mappe di anomaly score usate da evalAnomalyEoMT e genera le mappe MSP.
    '''
    logits_cpu = pixel_logits.detach().cpu()

    anomaly_result_logit = 1.0 - np.max(logits_cpu.numpy(), axis=0)
    
    probs_tensor = F.softmax(logits_cpu, dim=0)
    anomaly_result_softmax = 1.0 - np.max(probs_tensor.numpy(), axis=0)
    
    anomaly_result_entropy = -torch.sum(probs_tensor * torch.log(probs_tensor.clamp_min(1e-12)), dim=0).numpy()
    
    anomaly_result_rba = -torch.sum(torch.tanh(logits_cpu), dim=0).numpy()

    maps = {
        "logit": anomaly_result_logit,
        "softmax": anomaly_result_softmax,
        "entropy": anomaly_result_entropy,
        "rba": anomaly_result_rba,
    }

    if temperatures:
        for t in temperatures:
            logits_t = logits_cpu / t
            probs_t = F.softmax(logits_t, dim=0)
            anomaly_t = 1.0 - np.max(probs_t.numpy(), axis=0)
            maps[f"msp_T_{t}"] = anomaly_t

    return maps


def normalize_map(score_map):
    '''
    Normalizza una mappa numerica in [0, 1] per renderla come overlay grafico.
    '''
    score_map = score_map.astype(np.float32)
    min_value = np.nanmin(score_map)
    max_value = np.nanmax(score_map)

    if np.isclose(max_value, min_value):
        return np.zeros_like(score_map, dtype=np.float32)

    return (score_map - min_value) / (max_value - min_value)


def resize_original_for_plot(original_image, size=(1024, 1024)):
    '''
    Ridimensiona l'immagine originale alla risoluzione delle mappe predette.
    '''
    resized = original_image.resize(size, Image.BILINEAR)
    return np.array(resized)


def colorize_semantic_prediction(prediction):
    '''
    Converte una mappa di classi Cityscapes in una immagine RGB.
    '''
    colored_prediction = np.zeros((*prediction.shape, 3), dtype=np.uint8)

    for class_id, color in enumerate(CITYSCAPES_PALETTE):
        colored_prediction[prediction == class_id] = color

    return colored_prediction


def compute_semantic_prediction_and_probabilities(pixel_logits):
    '''
    Ricava predizione semantica, probabilita' e confidence dai logits EoMT.
    '''
    probabilities = F.softmax(pixel_logits.detach().cpu(), dim=0)
    prediction = torch.argmax(probabilities, dim=0).numpy().astype(np.uint8)
    confidence = torch.max(probabilities, dim=0).values.numpy()
    
    return prediction, probabilities.numpy(), confidence


def summarize_predicted_regions(prediction, probabilities, min_region_pixels=500, max_regions=30):
    '''
    Riassume le regioni connesse predette dal modello in righe da salvare su CSV.
    '''
    region_rows = []
    structure = np.ones((3, 3), dtype=np.uint8)

    for class_id in np.unique(prediction):
        class_mask = prediction == class_id
        labeled_regions, num_regions = ndimage.label(class_mask, structure=structure)

        for region_id in range(1, num_regions + 1):
            region_mask = labeled_regions == region_id
            area_pixels = int(region_mask.sum())

            if area_pixels < min_region_pixels:
                continue

            mean_probabilities = probabilities[:, region_mask].mean(axis=1)
            sorted_ids = np.argsort(mean_probabilities)[::-1]
            
            top5 = "; ".join(
                f"{CITYSCAPES_CLASSES[idx]}={mean_probabilities[idx]:.4f}"
                for idx in sorted_ids[:5]
            )

            row = {
                "predicted_class_id": int(class_id),
                "predicted_class_name": CITYSCAPES_CLASSES[int(class_id)],
                "area_pixels": area_pixels,
                "mean_confidence": float(mean_probabilities[int(class_id)]),
                "top5_mean_probabilities": top5,
            }

            for idx, class_name in enumerate(CITYSCAPES_CLASSES):
                safe_class_name = class_name.replace(" ", "_")
                row[f"prob_{idx:02d}_{safe_class_name}"] = float(mean_probabilities[idx])

            region_rows.append(row)

    region_rows.sort(key=lambda row: row["area_pixels"], reverse=True)
    return region_rows[:max_regions]


def save_region_probability_csv(region_rows, output_csv_path):
    '''
    Salva su CSV le probabilita' medie delle regioni predette.
    '''
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)

    if not region_rows:
        with open(output_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["message"])
            writer.writerow(["No predicted regions passed the min_region_pixels threshold."])
        return

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(region_rows[0].keys()))
        writer.writeheader()
        writer.writerows(region_rows)


def save_anomaly_visualization(image_path, output_path, score_name, score_map, original_image):
    '''
    Crea una figura PNG per una mappa anomaly (allineato alla logica ERFNet).
    '''
    normalized_score = normalize_map(score_map)
    original_np = resize_original_for_plot(original_image, size=(score_map.shape[1], score_map.shape[0]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(original_np)
    axes[0].set_title("Original Image")

    heatmap = axes[1].imshow(score_map, cmap="hot")
    axes[1].set_title(f"Anomaly Score {score_name}")
    fig.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(original_np)
    axes[2].imshow(normalized_score, cmap="hot", alpha=0.45)
    axes[2].set_title(f"Overlay {score_name}")

    for ax in axes:
        ax.axis("off")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def save_semantic_prediction_visualization(
    image_path,
    output_path,
    probability_csv_path,
    original_image,
    prediction,
    probabilities,
    confidence,
    min_region_pixels=500,
    max_regions=30,
):
    '''
    Salva la visualizzazione semantica e il CSV delle probabilita' (allineato alla logica ERFNet).
    '''
    colored_prediction = colorize_semantic_prediction(prediction)
    present_classes = np.unique(prediction)

    legend_handles = [
        Patch(
            facecolor=CITYSCAPES_PALETTE[class_id] / 255.0,
            edgecolor="black",
            label=f"{class_id}: {CITYSCAPES_CLASSES[class_id]}",
        )
        for class_id in present_classes
    ]

    original_np = resize_original_for_plot(original_image, size=(prediction.shape[1], prediction.shape[0]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(original_np)
    axes[0].set_title("Original Image")

    axes[1].imshow(colored_prediction)
    axes[1].set_title("EoMT Predicted Classes")
    axes[1].legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=3,
        fontsize=7,
        frameon=False,
    )

    conf_plot = axes[2].imshow(confidence, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[2].set_title("Predicted Class Confidence")
    fig.colorbar(conf_plot, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    region_rows = summarize_predicted_regions(
        prediction=prediction,
        probabilities=probabilities,
        min_region_pixels=min_region_pixels,
        max_regions=max_regions,
    )
    save_region_probability_csv(region_rows, probability_csv_path)


def load_anomaly_ground_truth(image_path):
    '''
    Carica la ground truth anomaly corrispondente a una immagine.
    '''
    path_gt = create_pathGT(image_path)
    mask = Image.open(path_gt)
    mask = target_transform(mask)
    return create_oodgts(mask, path_gt)


def create_empty_metric_storage(score_keys):
    '''
    Crea la struttura dati usata per accumulare ground truth e score anomaly.
    '''
    return {
        "ood_gts": [],
        "anomaly_scores": {k: [] for k in score_keys},
    }


def add_image_to_metric_storage(image_path, anomaly_score_maps, metric_storage):
    '''
    Aggiunge una immagine agli accumulatori delle metriche anomaly (con catch errori).
    '''
    try:
        ood_gts = load_anomaly_ground_truth(image_path)
    except FileNotFoundError:
        print("  Metriche saltate: ground truth non trovata.")
        return

    if 1 not in np.unique(ood_gts):
        print("  Metriche saltate: la ground truth non contiene anomalie.")
        return

    metric_storage["ood_gts"].append(ood_gts)

    for score_name, score_map in anomaly_score_maps.items():
        if score_name in metric_storage["anomaly_scores"]:
            metric_storage["anomaly_scores"][score_name].append(score_map)


def print_anomaly_metric_results(metric_storage):
    '''
    Calcola e stampa AUPRC e FPR@TPR95.
    '''
    if not metric_storage["ood_gts"]:
        print("Metriche anomaly non calcolate: nessuna immagine valutabile con ground truth anomala.")
        return

    for score_name in metric_storage["anomaly_scores"].keys():
        prc_auc, fpr = eval_score(
            metric_storage["ood_gts"],
            metric_storage["anomaly_scores"][score_name],
        )
        print(f"AUPRC {score_name} score: {prc_auc * 100.0:.2f}")
        print(f"FPR@TPR95 {score_name}: {fpr * 100.0:.2f}")


def build_output_paths(image_path, output_dir, score_keys):
    '''
    Costruisce i nomi dei file di output dinamici (allineato alla logica ERFNet).
    '''
    image_stem = Path(image_path).stem
    output_dir = Path(output_dir)
    
    paths = {
        "prediction": output_dir / f"{image_stem}_prediction.png",
        "probabilities": output_dir / f"{image_stem}_predicted_regions_probabilities.csv",
    }
    
    for key in score_keys:
        paths[key] = output_dir / f"{image_stem}_{key}.png"

    return paths


def visualize_single_image(
    image_path,
    model,
    device,
    output_dir="visualizations",
    mode="both",
    anomaly_score="rba",
    temperatures=None,
    min_region_pixels=500,
    max_regions=30,
):
    '''
    Esegue tutta la pipeline di visualizzazione per una singola immagine.
    '''
    original_image, pixel_logits = compute_pixel_logits_for_image(image_path, model, device)

    anomaly_score_maps = compute_anomaly_score_maps(pixel_logits, temperatures)
    output_paths = build_output_paths(image_path, output_dir, anomaly_score_maps.keys())
    prediction, probabilities, confidence = compute_semantic_prediction_and_probabilities(pixel_logits)

    if mode in ("anomaly", "both"):
        if anomaly_score == "all":
            score_names = list(anomaly_score_maps.keys())
        else:
            score_names = [anomaly_score]
            if temperatures:
                score_names.extend([f"msp_T_{t}" for t in temperatures])
        
        score_names = list(dict.fromkeys(score_names))
        
        for score_name in score_names:
            if score_name in anomaly_score_maps:
                save_anomaly_visualization(
                    image_path=image_path,
                    output_path=output_paths[score_name],
                    score_name=score_name,
                    score_map=anomaly_score_maps[score_name],
                    original_image=original_image,
                )

    if mode in ("prediction", "both"):
        save_semantic_prediction_visualization(
            image_path=image_path,
            output_path=output_paths["prediction"],
            probability_csv_path=output_paths["probabilities"],
            original_image=original_image,
            prediction=prediction,
            probabilities=probabilities,
            confidence=confidence,
            min_region_pixels=min_region_pixels,
            max_regions=max_regions,
        )

    del pixel_logits, prediction, probabilities, confidence
    if device == "cuda":
        torch.cuda.empty_cache()

    return output_paths, anomaly_score_maps


def expand_input_paths(input_patterns):
    '''
    Espande una lista di path o glob in una lista ordinata di immagini.
    '''
    image_paths = []

    for input_pattern in input_patterns:
        expanded = sorted(glob.glob(os.path.expanduser(str(input_pattern))))
        if expanded:
            image_paths.extend(expanded)
        elif Path(input_pattern).exists():
            image_paths.append(str(input_pattern))

    return sorted(dict.fromkeys(image_paths))


def main():
    '''
    Entry point da riga di comando per visualizzare immagini con EoMT.
    '''
    parser = ArgumentParser()
    parser.add_argument(
        "--input", 
        required=True, 
        nargs="+", 
        help="Path o glob delle immagini da visualizzare."
    )
    parser.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/ml_anomaly_segmentation/visualizations",
        help="Cartella in cui salvare PNG e CSV prodotti.",
    )
    parser.add_argument(
        "--config-path",
        default="configs/dinov2/cityscapes/semantic/eomt_base_640.yaml",
        help="Path della config EoMT.",
    )
    parser.add_argument(
        "--state-dict-path",
        default="/content/drive/MyDrive/ml_anomaly_segmentation/eomt_cityscapes.bin",
        help="Path del file .bin con i pesi del modello.",
    )
    parser.add_argument(
        "--mode",
        choices=["anomaly", "prediction", "both"],
        default="both",
        help="Tipo di visualizzazione da salvare.",
    )
    parser.add_argument(
        "--anomaly-score",
        choices=["logit", "softmax", "entropy", "rba", "all"],
        default="rba",
        help="Score anomaly da visualizzare quando mode e' anomaly/both.",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=None,
        help="Lista di temperature per calcolare e stampare le relative mappe MSP.",
    )
    parser.add_argument(
        "--min-region-pixels",
        type=int,
        default=500,
        help="Area minima per includere una regione nel CSV delle probabilità.",
    )
    parser.add_argument(
        "--max-regions",
        type=int,
        default=30,
        help="Numero massimo di regioni da salvare nel CSV per ogni immagine.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device da usare. Se omesso, usa CPU.",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Non cerca le ground truth e non calcola AUPRC/FPR.",
    )
    args = parser.parse_args()

    model, device = load_eomt_for_visualization(
        config_path=args.config_path,
        state_dict_path=args.state_dict_path,
        device=args.device,
    )

    image_paths = expand_input_paths(args.input)
    if not image_paths:
        raise FileNotFoundError(f"Nessuna immagine trovata con input: {args.input}")

    score_keys = ["logit", "softmax", "entropy", "rba"]
    if args.temperatures:
        score_keys.extend([f"msp_T_{t}" for t in args.temperatures])

    metric_storage = create_empty_metric_storage(score_keys)

    for image_path in image_paths:
        print(f"Visualizzo: {image_path}")
        output_paths, anomaly_score_maps = visualize_single_image(
            image_path=image_path,
            model=model,
            device=device,
            output_dir=args.output_dir,
            mode=args.mode,
            anomaly_score=args.anomaly_score,
            temperatures=args.temperatures,
            min_region_pixels=args.min_region_pixels,
            max_regions=args.max_regions,
        )

        if not args.skip_metrics:
            add_image_to_metric_storage(
                image_path=image_path,
                anomaly_score_maps=anomaly_score_maps,
                metric_storage=metric_storage,
            )

        if args.mode in ("anomaly", "both"):
            saved_scores = [args.anomaly_score] if args.anomaly_score != "all" else list(anomaly_score_maps.keys())
            if args.temperatures and args.anomaly_score != "all":
                saved_scores.extend([f"msp_T_{t}" for t in args.temperatures])
            
            for score_name in dict.fromkeys(saved_scores):
                if score_name in output_paths:
                    print(f"  {score_name} salvato in: {output_paths[score_name]}")

        if args.mode in ("prediction", "both"):
            print(f"  Predizione salvata in: {output_paths['prediction']}")
            print(f"  Probabilità regioni salvate in: {output_paths['probabilities']}")

    if not args.skip_metrics:
        print_anomaly_metric_results(metric_storage)


if __name__ == "__main__":
    main()