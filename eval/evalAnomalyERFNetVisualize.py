import csv
import glob
import os
import random
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.nn import functional as F
from torchvision.transforms import Compose, Resize, ToTensor

from erfnet import ERFNet
from functions import create_oodgts, create_pathGT, eval_score, load_my_state_dict
from transform import colormap_cityscapes


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


SEED = 42
NUM_CLASSES = 20
IGNORE_CLASS_ID = 19


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
    "ignore",
]


CITYSCAPES_PALETTE = colormap_cityscapes(NUM_CLASSES)


input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
    ]
)


target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


def set_reproducibility(seed=SEED):
    """
    Imposta i seed di Python, NumPy e PyTorch per rendere l'inferenza ripetibile.

    Viene usata prima di caricare il modello, cosi eventuali operazioni
    non deterministiche della pipeline partono sempre dallo stesso stato.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def resolve_default_weights_dir():
    """
    Restituisce la cartella dei pesi ERFNet inclusa nel progetto.

    Il path viene calcolato a partire dalla posizione di questo file, quindi lo
    script funziona anche se viene lanciato da una working directory diversa.
    """
    return Path(__file__).resolve().parents[1] / "trained_models"


def load_erfnet_for_visualization(load_dir=None, load_weights="erfnet_pretrained.pth", device=None):
    """
    Carica ERFNet nello stesso modo degli script di valutazione in eval.
    """
    set_reproducibility()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    load_dir = Path(load_dir) if load_dir is not None else resolve_default_weights_dir()
    weightspath = load_dir / load_weights

    print(f"Loading weights: {weightspath}")

    model = ERFNet(NUM_CLASSES)
    if device == "cuda":
        model = torch.nn.DataParallel(model).cuda()

    state_dict = torch.load(str(weightspath), map_location=lambda storage, loc: storage)
    model = load_my_state_dict(model, state_dict)
    model.eval()

    print("Model and weights LOADED successfully")
    return model, device


def load_image_like_eval_anomaly(image_path, device):
    """
    Carica e prepara una singola immagine nello stesso formato di evalAnomalyERFNet.

    L'immagine viene convertita in RGB, ridimensionata a 512x1024, trasformata in
    tensore PyTorch e spostata sul device scelto. Ritorna anche l'immagine PIL
    originale, utile per costruire gli overlay salvati su disco.
    """
    original_image = Image.open(image_path).convert("RGB")
    image_tensor = input_transform(original_image).unsqueeze(0).float().to(device)
    return original_image, image_tensor


def compute_logits_for_image(image_path, model, device):
    """
    Esegue l'inferenza ERFNet su una immagine e restituisce i logits pixel-wise.

    I logits hanno forma `num_classi x altezza x larghezza` dopo aver rimosso la
    dimensione batch. Sono la base comune per calcolare segmentazione, confidence
    e mappe di anomaly score.
    """
    original_image, image_tensor = load_image_like_eval_anomaly(image_path, device)

    with torch.no_grad():
        logits = model(image_tensor).squeeze(0)

    return original_image, logits


def compute_anomaly_score_maps(logits, temperatures=None):
    """
    Calcola le mappe di anomalia usate dagli script di valutazione ERFNet.

    Produce tre score post-hoc: max-logit (`1 - max(logit)`), MSP
    (`1 - max(softmax)`) ed entropia della softmax. Valori piu' alti indicano
    pixel che il modello considera piu' incerti o sospetti.
    
    Se passato in input il vettore delle temperature calcola MSP t
    """
    logits_cpu = logits.detach().cpu()

    anomaly_result_logit = 1.0 - np.max(logits_cpu.numpy(), axis=0)
    probs_tensor = F.softmax(logits_cpu, dim=0)
    anomaly_result_softmax = 1.0 - np.max(probs_tensor.numpy(), axis=0)
    anomaly_result_entropy = -torch.sum(probs_tensor * torch.log(probs_tensor.clamp_min(1e-12)), dim=0).numpy()

    maps = {
        "logit": anomaly_result_logit,
        "softmax": anomaly_result_softmax,
        "entropy": anomaly_result_entropy,
    }

    if temperatures:
        for t in temperatures:
            logits_t = logits_cpu / t
            probs_t = F.softmax(logits_t, dim=0)
            anomaly_t = 1.0 - np.max(probs_t.numpy(), axis=0)
            maps[f"msp_T_{t}"] = anomaly_t

    return maps


def normalize_map(score_map):
    """
    Normalizza una mappa numerica in [0, 1] per renderla come overlay grafico.

    La normalizzazione e' solo visuale: non viene usata per calcolare metriche e
    non modifica gli anomaly score originali.
    """
    score_map = score_map.astype(np.float32)
    min_value = np.nanmin(score_map)
    max_value = np.nanmax(score_map)

    if np.isclose(max_value, min_value):
        return np.zeros_like(score_map, dtype=np.float32)

    return (score_map - min_value) / (max_value - min_value)


def resize_original_for_plot(original_image, size=(1024, 512)):
    """
    Ridimensiona l'immagine originale alla risoluzione delle mappe predette.

    Serve per sovrapporre correttamente immagine RGB, heatmap anomaly e
    segmentazione ERFNet nelle figure salvate.
    """
    resized = original_image.resize(size, Image.BILINEAR)
    return np.array(resized)


def colorize_semantic_prediction(prediction):
    """
    Converte una mappa di classi ERFNet in una immagine RGB Cityscapes.

    Ogni indice di classe viene mappato al colore definito in `transform.py`, in
    modo coerente con le altre visualizzazioni Cityscapes del progetto.
    """
    colored_prediction = np.zeros((*prediction.shape, 3), dtype=np.uint8)

    for class_id, color in enumerate(CITYSCAPES_PALETTE):
        colored_prediction[prediction == class_id] = color

    return colored_prediction


def compute_semantic_prediction_and_probabilities(logits):
    """
    Ricava predizione semantica, probabilita' e confidence dai logits ERFNet.

    La predizione e' l'argmax della softmax per pixel; la confidence e' la
    probabilita' della classe vincente. Le probabilita' complete vengono
    mantenute per calcolare le statistiche medie sulle regioni connesse.
    """
    probabilities = F.softmax(logits.detach().cpu(), dim=0)
    prediction = torch.argmax(probabilities, dim=0).numpy().astype(np.uint8)
    confidence = torch.max(probabilities, dim=0).values.numpy()
    return prediction, probabilities.numpy(), confidence


def summarize_predicted_regions(prediction, probabilities, min_region_pixels=500, max_regions=30):
    """
    Riassume le regioni connesse predette dal modello in righe da salvare su CSV.

    Per ogni classe presente nella predizione, trova le componenti connesse,
    scarta quelle troppo piccole e calcola la probabilita' media di tutte le
    classi Cityscapes dentro la regione. In questo modo si ottiene una stima
    delle probabilita' associate agli oggetti/aree visualizzate.
    """
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
    """
    Salva su CSV le probabilita' medie delle regioni predette.

    Ogni riga descrive una regione connessa: classe predetta, area, confidence
    media, top-5 classi piu' probabili e probabilita' completa per tutte le
    classi. Se non ci sono regioni sopra soglia, salva comunque un CSV minimale.
    """
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
    """
    Crea una figura PNG per una mappa anomaly di ERFNet.

    La figura contiene tre pannelli: immagine originale, heatmap dello score e
    overlay della heatmap sull'immagine. `score_name` viene usato nei titoli per
    distinguere logit, softmax ed entropy.
    """
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
    """
    Salva la visualizzazione semantica e il CSV delle probabilita' per regione.

    La figura contiene immagine originale, segmentazione ERFNet colorata e mappa
    di confidence della classe predetta. Alla fine richiama anche il riepilogo
    delle componenti connesse e lo esporta come CSV.
    """
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
    axes[1].set_title("ERFNet Predicted Classes")
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
    """
    Carica la ground truth anomaly corrispondente a una immagine.

    Usa le stesse utility degli script eval: `create_pathGT` ricava il path della
    maschera e `create_oodgts` la converte in una maschera binaria OoD/IND.
    """
    path_gt = create_pathGT(image_path)
    mask = Image.open(path_gt)
    mask = target_transform(mask)
    return create_oodgts(mask, path_gt)


def create_empty_metric_storage():
    """
    Crea la struttura dati usata per accumulare ground truth e score anomaly.

    Le liste vengono riempite immagine per immagine e poi passate a `eval_score`
    per calcolare AUPRC e FPR@TPR95 sui tre score disponibili.
    """
    return {
        "ood_gts": [],
        "anomaly_scores": {
            "logit": [],
            "softmax": [],
            "entropy": [],
        },
    }


def add_image_to_metric_storage(image_path, anomaly_score_maps, metric_storage):
    """
    Aggiunge una immagine agli accumulatori delle metriche anomaly.

    Se la ground truth non esiste o non contiene pixel anomali, l'immagine viene
    saltata come negli script di valutazione originali. In caso contrario,
    ground truth e score vengono accodati nello stesso ordine.
    """
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
        metric_storage["anomaly_scores"][score_name].append(score_map)


def print_anomaly_metric_results(metric_storage):
    """
    Calcola e stampa AUPRC e FPR@TPR95 per logit, softmax ed entropy.

    Se nessuna immagine valida e' stata accumulata, stampa un messaggio esplicito
    invece di chiamare le metriche su liste vuote.
    """
    if not metric_storage["ood_gts"]:
        print("Metriche anomaly non calcolate: nessuna immagine valutabile con ground truth anomala.")
        return

    for score_name in ("logit", "softmax", "entropy"):
        prc_auc, fpr = eval_score(
            metric_storage["ood_gts"],
            metric_storage["anomaly_scores"][score_name],
        )
        print(f"AUPRC {score_name} score: {prc_auc * 100.0}")
        print(f"FPR@TPR95 {score_name}: {fpr * 100.0}")


def build_output_paths(image_path, output_dir):
    """
    Costruisce i nomi dei file di output per una immagine.

    Usa lo stem dell'immagine di input per generare path stabili per heatmap,
    predizione semantica e CSV delle probabilita'.
    """
    image_stem = Path(image_path).stem
    output_dir = Path(output_dir)

    return {
        "logit": output_dir / f"{image_stem}_logit.png",
        "softmax": output_dir / f"{image_stem}_softmax.png",
        "entropy": output_dir / f"{image_stem}_entropy.png",
        "prediction": output_dir / f"{image_stem}_prediction.png",
        "probabilities": output_dir / f"{image_stem}_predicted_regions_probabilities.csv",
    }


def visualize_single_image(
    image_path,
    model,
    device,
    output_dir="visualizations_erfnet",
    mode="both",
    anomaly_score="softmax",
    temperatures=None,
    min_region_pixels=500,
    max_regions=30,
):
    """
    Esegue tutta la pipeline di visualizzazione per una singola immagine.

    Fa una sola inferenza ERFNet, riusa gli stessi logits per anomaly score,
    segmentazione e confidence, salva gli output richiesti da `mode` e ritorna i
    path creati insieme alle mappe anomaly per eventuale valutazione aggregata.
    """
    output_paths = build_output_paths(image_path, output_dir)
    original_image, logits = compute_logits_for_image(image_path, model, device)

    anomaly_score_maps = compute_anomaly_score_maps(logits)
    prediction, probabilities, confidence = compute_semantic_prediction_and_probabilities(logits)

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

    del logits, prediction, probabilities, confidence
    if device == "cuda":
        torch.cuda.empty_cache()

    return output_paths, anomaly_score_maps


def expand_input_paths(input_patterns):
    """
    Espande una lista di path o glob in una lista ordinata di immagini.

    Accetta sia pattern come `images/*.jpg` sia file singoli. I duplicati vengono
    rimossi preservando un ordinamento stabile.
    """
    image_paths = []

    for input_pattern in input_patterns:
        expanded = sorted(glob.glob(os.path.expanduser(str(input_pattern))))
        if expanded:
            image_paths.extend(expanded)
        elif Path(input_pattern).exists():
            image_paths.append(str(input_pattern))

    return sorted(dict.fromkeys(image_paths))


def main():
    """
    Entry point da riga di comando per visualizzare immagini con ERFNet.

    Legge gli argomenti CLI, carica il modello, espande gli input, processa le
    immagini una alla volta e, se non richiesto diversamente, calcola anche le
    metriche anomaly aggregate sulle immagini con ground truth valida.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="Path o glob delle immagini da visualizzare, per esempio 'RoadAnomaly21/images/*.jpg'.",
    )
    parser.add_argument(
        "--output-dir",
        default="visualizations_erfnet",
        help="Cartella in cui salvare PNG e CSV prodotti.",
    )
    parser.add_argument(
        "--loadDir",
        default=str(resolve_default_weights_dir()),
        help="Cartella che contiene i pesi ERFNet.",
    )
    parser.add_argument("--loadWeights", default="erfnet_pretrained.pth")
    parser.add_argument(
        "--mode",
        choices=["anomaly", "prediction", "both"],
        default="both",
        help="Tipo di visualizzazione da salvare.",
    )
    parser.add_argument(
        "--anomaly-score",
        choices=["logit", "softmax", "entropy", "all"],
        default="softmax",
        help="Score anomaly da visualizzare quando mode e' anomaly/both.",
    )
    parser.add_argument(
        "--min-region-pixels",
        type=int,
        default=500,
        help="Area minima per includere una regione nel CSV delle probabilita'.",
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
        default=None,
        help="Device da usare. Se omesso, usa CUDA quando disponibile.",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Non cerca le ground truth e non calcola AUPRC/FPR.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Hai richiesto --device cuda, ma CUDA non e' disponibile.")

    model, device = load_erfnet_for_visualization(
        load_dir=args.loadDir,
        load_weights=args.loadWeights,
        device=args.device,
    )

    image_paths = expand_input_paths(args.input)
    if not image_paths:
        raise FileNotFoundError(f"Nessuna immagine trovata con input: {args.input}")

    score_keys = ["logit", "softmax", "entropy"]
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
            # Determina i file effettivamente generati
            saved_scores = [args.anomaly_score] if args.anomaly_score != "all" else list(anomaly_score_maps.keys())
            if args.temperatures and args.anomaly_score != "all":
                saved_scores.extend([f"msp_T_{t}" for t in args.temperatures])
            
            for score_name in dict.fromkeys(saved_scores):
                if score_name in output_paths:
                    print(f"  {score_name} salvato in: {output_paths[score_name]}")

        if args.mode in ("prediction", "both"):
            print(f"  Predizione salvata in: {output_paths['prediction']}")
            print(f"  Probabilita' regioni salvate in: {output_paths['probabilities']}")

    if not args.skip_metrics:
        print_anomaly_metric_results(metric_storage)


if __name__ == "__main__":
    main()
