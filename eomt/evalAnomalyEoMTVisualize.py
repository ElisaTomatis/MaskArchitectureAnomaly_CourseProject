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

    La funzione tiene separati `config_path` e `state_dict_path` per permettere di
    scegliere da riga di comando sia la configurazione sia il file `.bin` dei pesi.
    Uso una stringa `"cuda"`/`"cpu"` come device, per restare compatibile con i
    controlli presenti in `functions.load_model`.
    '''
    # Fisso il seed come negli script di valutazione, così la pipeline resta riproducibile.
    seed_everything(0, verbose=False)

    # Scelgo automaticamente CUDA quando disponibile, altrimenti CPU.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Leggo la configurazione EoMT, che contiene classi Python e iperparametri del modello.
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Nascondo il warning già gestito anche negli altri file di valutazione del progetto.
    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )

    # Costruisco il modello e carico i pesi `.bin` tramite la funzione condivisa.
    model = load_model(device, config, state_dict_path)

    # Ritorno anche il device, così le funzioni successive non devono ricostruirlo.
    return model, device


def load_image_like_eval_anomaly(image_path, device):
    '''
    Carica una immagine nello stesso formato usato da `evalAnomalyEoMT.py`.

    L'immagine viene convertita in RGB, ridimensionata a 1024x1024, trasformata in
    tensore e riportata su scala 0-255 `uint8`, perché `window_imgs_semantic`
    ricostruisce internamente immagini PIL a partire dal tensore.
    '''
    # Apro l'immagine originale in RGB, evitando problemi con immagini in grayscale o RGBA.
    original_image = Image.open(image_path).convert("RGB")

    # Applico la stessa trasformazione usata negli script di valutazione.
    image_tensor = input_transform(original_image).float()

    # Riporto il tensore su scala 0-255 e tipo uint8, come in `evalAnomalyEoMT.py`.
    image_tensor = (image_tensor * 255).to(torch.uint8)

    # Sposto l'immagine sul device del modello solo dopo aver preservato l'originale PIL.
    image_tensor = image_tensor.to(device)

    # Ritorno sia l'immagine originale sia il tensore pronto per EoMT.
    return original_image, image_tensor


def compute_pixel_logits_for_image(image_path, model, device):
    '''
    Esegue l'inferenza EoMT su una singola immagine e restituisce i logits pixel-wise.

    Questa funzione passa davvero dal modello EoMT e usa `compute_logits`, quindi
    include tutta la logica dei crop/finestrature (`window_imgs_semantic`) e della
    ricomposizione (`revert_window_logits_semantic`).
    '''
    # Carico l'immagine con la stessa pipeline degli script di valutazione anomalie.
    original_image, image_tensor = load_image_like_eval_anomaly(image_path, device)

    # Disabilito il calcolo dei gradienti perché qui facciamo solo inferenza/visualizzazione.
    with torch.no_grad():
        # `compute_logits` si aspetta una lista/batch di immagini: qui ne passiamo una sola.
        pixel_logits = compute_logits([image_tensor], device, model)

    # Ritorno l'immagine originale e i logits, mantenendo i logits su device per eventuali elaborazioni.
    return original_image, pixel_logits


def compute_rba_anomaly_score(pixel_logits):
    '''
    Calcola lo score di anomalia RbA a partire dai logits pixel-wise.

    La formula è la stessa usata in `evalAnomalyEoMT.py`:
    `RbA = - sum_c tanh(logit_c)`. Valori più alti indicano pixel più sospetti.
    '''
    # Porto i logits su CPU e stacco il grafo per trasformarli in una mappa NumPy.
    logits_cpu = pixel_logits.detach().cpu()

    # Applico esattamente la formula RbA usata nello script di valutazione.
    anomaly_score_rba = -torch.sum(torch.tanh(logits_cpu), dim=0).numpy()

    # Restituisco una matrice HxW pronta per essere visualizzata con matplotlib.
    return anomaly_score_rba


def compute_anomaly_score_maps(pixel_logits):
    '''
    Calcola le quattro mappe di anomaly score usate da `evalAnomalyEoMT.py`.

    Le formule sono mantenute identiche allo script di valutazione originale:
    logit, softmax, entropy e RbA vengono calcolati sui logits pixel-wise
    restituiti dal modello EoMT.
    '''
    # Porto i logits su CPU una sola volta, cosi tutte le metriche usano gli stessi valori.
    logits_cpu = pixel_logits.detach().cpu()

    # Score logit: uno meno il massimo logit tra le classi per ogni pixel.
    anomaly_result_logit = 1.0 - np.max(logits_cpu.numpy(), axis=0)

    # Score softmax: uno meno la massima probabilita softmax per pixel.
    probs_tensor = F.softmax(logits_cpu, dim=0)
    anomaly_result_softmax = 1.0 - np.max(probs_tensor.numpy(), axis=0)

    # Score entropy: entropia della distribuzione softmax pixel-wise.
    anomaly_result_entropy = -torch.sum(probs_tensor * torch.log(probs_tensor), dim=0).numpy()

    # Score RbA: stessa formula gia usata per la heatmap visuale.
    anomaly_result_rba = -torch.sum(torch.tanh(logits_cpu), dim=0).numpy()

    # Ritorno un dizionario, cosi il chiamante puo accumulare le metriche per nome.
    return {
        "logit": anomaly_result_logit,
        "softmax": anomaly_result_softmax,
        "entropy": anomaly_result_entropy,
        "rba": anomaly_result_rba,
    }


def load_anomaly_ground_truth(image_path):
    '''
    Carica la maschera ground-truth OOD associata a una immagine.

    Il path e la conversione della maschera sono gli stessi usati in
    `evalAnomalyEoMT.py`, cosi le metriche finali restano confrontabili con lo
    script di valutazione originale.
    '''
    # Ricavo il path della maschera a partire dal path dell'immagine.
    path_gt = create_pathGT(image_path)

    # Ridimensiono la maschera con nearest-neighbor, preservando gli ID delle classi.
    mask = Image.open(path_gt)
    mask = target_transform(mask)

    # Converto la maschera nel formato binario OOD/IND usato da `eval_score`.
    return create_oodgts(mask, path_gt)


def create_empty_metric_storage():
    '''
    Prepara le liste in cui accumulare ground truth e anomaly score.

    La struttura rispecchia le quattro metriche di `evalAnomalyEoMT.py`, ma usa
    un dizionario per tenere il codice piu compatto e leggibile.
    '''
    # Ogni chiave contiene le mappe score delle immagini valutabili.
    anomaly_scores = {
        "logit": [],
        "softmax": [],
        "entropy": [],
        "rba": [],
    }

    # La ground truth resta una lista separata, condivisa da tutte le metriche.
    return {
        "ood_gts": [],
        "anomaly_scores": anomaly_scores,
    }


def add_image_to_metric_storage(image_path, anomaly_score_maps, metric_storage):
    '''
    Aggiunge una immagine alle liste usate per calcolare AUPRC e FPR@TPR95.

    Come in `evalAnomalyEoMT.py`, le immagini senza pixel anomali vengono scartate
    dalle metriche, perche non contribuiscono alla valutazione OOD.
    '''
    # Carico la ground truth OOD corrispondente all'immagine appena visualizzata.
    ood_gts = load_anomaly_ground_truth(image_path)

    # Mantengo lo stesso comportamento dello script originale: scarto immagini senza anomalie.
    if 1 not in np.unique(ood_gts):
        print("  Metriche saltate: la ground truth non contiene anomalie.")
        return

    # Accumulo la maschera ground truth una sola volta.
    metric_storage["ood_gts"].append(ood_gts)

    # Accumulo tutte le mappe score nello stesso ordine della ground truth.
    for score_name, score_map in anomaly_score_maps.items():
        metric_storage["anomaly_scores"][score_name].append(score_map)


def print_anomaly_metric_results(metric_storage):
    '''
    Calcola e stampa AUPRC e FPR@TPR95 per tutte le anomaly score map.

    L'output a terminale mantiene lo stesso formato di `evalAnomalyEoMT.py`: se
    e stata valutata una sola immagine, i valori sono relativi solo a quella;
    altrimenti sono calcolati sull'insieme di tutte le immagini accumulate.
    '''
    # Se nessuna immagine ha una ground truth valutabile, non posso calcolare le metriche.
    if not metric_storage["ood_gts"]:
        print("Metriche anomaly non calcolate: nessuna immagine contiene anomalie nella ground truth.")
        return

    # Calcolo le metriche con la stessa funzione condivisa dello script eval.
    prc_auc_logit, fpr_logit = eval_score(
        metric_storage["ood_gts"],
        metric_storage["anomaly_scores"]["logit"],
    )
    prc_auc_softmax, fpr_softmax = eval_score(
        metric_storage["ood_gts"],
        metric_storage["anomaly_scores"]["softmax"],
    )
    prc_auc_entropy, fpr_entropy = eval_score(
        metric_storage["ood_gts"],
        metric_storage["anomaly_scores"]["entropy"],
    )
    prc_auc_rba, fpr_rba = eval_score(
        metric_storage["ood_gts"],
        metric_storage["anomaly_scores"]["rba"],
    )

    # Stampo i risultati nello stesso ordine e con le stesse etichette di `evalAnomalyEoMT.py`.
    print(f"AUPRC logit score: {prc_auc_logit * 100.0}")
    print(f"FPR@TPR95 logit: {fpr_logit * 100.0}")

    print(f"AUPRC softmax score: {prc_auc_softmax * 100.0}")
    print(f"FPR@TPR95 softmax: {fpr_softmax * 100.0}")

    print(f"AUPRC entropy score: {prc_auc_entropy * 100.0}")
    print(f"FPR@TPR95 entropy: {fpr_entropy * 100.0}")

    print(f"AUPRC rba score: {prc_auc_rba * 100.0}")
    print(f"FPR@TPR95 rba: {fpr_rba * 100.0}")


def normalize_map(score_map):
    '''
    Normalizza una mappa numerica nell'intervallo [0, 1] per usarla come overlay.

    La normalizzazione è solo grafica: non modifica lo score RbA originale e non
    viene usata per metriche o decisioni quantitative.
    '''
    # Converto a float32 per evitare cast impliciti durante la normalizzazione.
    score_map = score_map.astype(np.float32)

    # Calcolo min e max ignorando eventuali valori non finiti.
    min_value = np.nanmin(score_map)
    max_value = np.nanmax(score_map)

    # Evito divisioni per zero quando la mappa è costante.
    if np.isclose(max_value, min_value):
        return np.zeros_like(score_map, dtype=np.float32)

    # Porto la mappa nell'intervallo [0, 1].
    return (score_map - min_value) / (max_value - min_value)


def resize_original_for_plot(original_image, size=(1024, 1024)):
    '''
    Ridimensiona l'immagine originale alla dimensione usata dal modello.

    Questo rende sovrapponibili immagine, heatmap RbA e segmentazione predetta.
    '''
    # Matplotlib lavora comodamente con array NumPy RGB.
    resized = original_image.resize(size, Image.BILINEAR)

    # Converto da PIL a NumPy mantenendo valori RGB 0-255.
    return np.array(resized)


def save_rba_anomaly_visualization(
    image_path,
    model,
    device,
    output_path,
    pixel_logits=None,
):
    '''
    Salva una figura con immagine originale, heatmap RbA e overlay RbA.

    Se `pixel_logits` viene passato dall'esterno, la funzione lo riusa per evitare
    una seconda inferenza sulla stessa immagine. Altrimenti carica l'immagine,
    passa dal modello e calcola i logits con tutta la pipeline dei crop.
    '''
    # Se i logits non sono già disponibili, eseguo l'inferenza completa sul modello.
    if pixel_logits is None:
        original_image, pixel_logits = compute_pixel_logits_for_image(image_path, model, device)
    else:
        original_image = Image.open(image_path).convert("RGB")

    # Calcolo la mappa RbA usando la stessa formula dello script di valutazione.
    rba_score = compute_rba_anomaly_score(pixel_logits)

    # Creo una versione normalizzata solo per l'overlay visivo.
    normalized_rba = normalize_map(rba_score)

    # Preparo l'immagine RGB originale nella stessa dimensione dei logits.
    original_np = resize_original_for_plot(original_image, size=(rba_score.shape[1], rba_score.shape[0]))

    # Creo una figura a tre pannelli: input, score puro, overlay leggibile.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Primo pannello: immagine originale ridimensionata come vista dal modello.
    axes[0].imshow(original_np)
    axes[0].set_title("Immagine originale")

    # Secondo pannello: heatmap RbA con colorbar, utile per leggere valori relativi.
    heatmap = axes[1].imshow(rba_score, cmap="hot")
    axes[1].set_title("Score anomalia RbA")
    fig.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)

    # Terzo pannello: overlay RbA sull'immagine, più facile da interpretare visivamente.
    axes[2].imshow(original_np)
    axes[2].imshow(normalized_rba, cmap="hot", alpha=0.45)
    axes[2].set_title("Overlay RbA")

    # Rimuovo gli assi da tutti i pannelli per produrre immagini pulite.
    for ax in axes:
        ax.axis("off")

    # Creo la cartella di destinazione se non esiste.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Salvo la figura finale su disco.
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    # Ritorno score e logits per eventuale riuso o debugging.
    return rba_score, pixel_logits


def colorize_semantic_prediction(prediction):
    '''
    Trasforma una maschera di classi Cityscapes in una immagine RGB colorata.

    Ogni ID di classe 0-18 viene mappato sulla palette standard Cityscapes, così
    la visualizzazione è stabile e confrontabile tra immagini diverse.
    '''
    # Creo un'immagine RGB vuota con la stessa altezza/larghezza della predizione.
    colored_prediction = np.zeros((*prediction.shape, 3), dtype=np.uint8)

    # Assegno il colore Cityscapes corrispondente a ogni classe predetta.
    for class_id, color in enumerate(CITYSCAPES_PALETTE):
        colored_prediction[prediction == class_id] = color

    # Restituisco la maschera RGB pronta per matplotlib o PIL.
    return colored_prediction


def compute_semantic_prediction_and_probabilities(pixel_logits):
    '''
    Calcola predizione semantica, softmax e confidence per pixel.

    Le probabilità sono pixel-wise, perché EoMT semantic produce logits per ogni
    classe e per ogni pixel. Le statistiche per regione vengono costruite dopo
    aggregando queste probabilità sui pixel della stessa regione predetta.
    '''
    # Calcolo la softmax lungo la dimensione delle classi.
    probabilities = F.softmax(pixel_logits.detach().cpu(), dim=0)

    # La classe predetta è l'argmax della probabilità per ogni pixel.
    prediction = torch.argmax(probabilities, dim=0).numpy().astype(np.uint8)

    # La confidence è la probabilità della classe vincente in ogni pixel.
    confidence = torch.max(probabilities, dim=0).values.numpy()

    # Ritorno anche le probabilità complete, utili per le analisi per regione.
    return prediction, probabilities.numpy(), confidence


def summarize_predicted_regions(
    prediction,
    probabilities,
    min_region_pixels=500,
    max_regions=30,
):
    '''
    Aggrega le probabilità pixel-wise su regioni connesse predette dal modello.

    Questa è la risposta più fedele alla domanda sulle "probabilità degli oggetti":
    il modello semantic non produce istanze/oggetti, quindi identifichiamo regioni
    connesse della stessa classe predetta e calcoliamo la distribuzione media delle
    probabilità Cityscapes dentro ciascuna regione.
    '''
    # Preparo la struttura in cui accumulare le statistiche delle regioni trovate.
    region_rows = []

    # Uso connettività 8-neighbor per considerare connessi pixel che si toccano anche in diagonale.
    structure = np.ones((3, 3), dtype=np.uint8)

    # Analizzo separatamente ogni classe Cityscapes presente nella predizione.
    for class_id in np.unique(prediction):
        # Creo una maschera binaria per la classe corrente.
        class_mask = prediction == class_id

        # Etichetto le componenti connesse della classe corrente.
        labeled_regions, num_regions = ndimage.label(class_mask, structure=structure)

        # Scorro tutte le componenti connesse individuate da scipy.
        for region_id in range(1, num_regions + 1):
            # Isolo i pixel della regione corrente.
            region_mask = labeled_regions == region_id

            # Calcolo l'area in pixel della regione.
            area_pixels = int(region_mask.sum())

            # Scarto regioni molto piccole, spesso rumore o dettagli non leggibili.
            if area_pixels < min_region_pixels:
                continue

            # Calcolo la probabilità media di ciascuna classe dentro la regione.
            mean_probabilities = probabilities[:, region_mask].mean(axis=1)

            # Ordino le classi dalla più probabile alla meno probabile nella regione.
            sorted_ids = np.argsort(mean_probabilities)[::-1]

            # Preparo una sintesi top-5 leggibile, oltre alle 19 probabilità complete.
            top5 = "; ".join(
                f"{CITYSCAPES_CLASSES[idx]}={mean_probabilities[idx]:.4f}"
                for idx in sorted_ids[:5]
            )

            # Aggiungo una riga con metadati e distribuzione completa.
            row = {
                "predicted_class_id": int(class_id),
                "predicted_class_name": CITYSCAPES_CLASSES[int(class_id)],
                "area_pixels": area_pixels,
                "mean_confidence": float(mean_probabilities[int(class_id)]),
                "top5_mean_probabilities": top5,
            }

            # Inserisco anche una colonna per ciascuna classe, utile per analisi successive.
            for idx, class_name in enumerate(CITYSCAPES_CLASSES):
                row[f"prob_{idx:02d}_{class_name.replace(' ', '_')}"] = float(mean_probabilities[idx])

            # Salvo la riga nella lista complessiva.
            region_rows.append(row)

    # Ordino le regioni per area decrescente, così le componenti più importanti vengono prima.
    region_rows.sort(key=lambda row: row["area_pixels"], reverse=True)

    # Limito il numero di regioni salvate per non produrre CSV enormi su immagini rumorose.
    return region_rows[:max_regions]


def save_region_probability_csv(region_rows, output_csv_path):
    '''
    Salva su CSV le probabilità medie per regione predetta.

    Il CSV contiene una riga per ogni regione connessa sopra soglia e una colonna
    per ciascuna delle 19 classi Cityscapes, così puoi verificare se una zona
    anomala viene predetta con alta sicurezza o con probabilità più uniformi.
    '''
    # Creo la cartella di output prima di aprire il file.
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)

    # Se non ci sono regioni sopra soglia, salvo comunque un CSV minimale.
    if not region_rows:
        with open(output_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["message"])
            writer.writerow(["No predicted regions passed the min_region_pixels threshold."])
        return

    # Ricavo l'header dalle chiavi della prima riga.
    fieldnames = list(region_rows[0].keys())

    # Scrivo tutte le righe in formato CSV.
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(region_rows)


def save_semantic_prediction_visualization(
    image_path,
    model,
    device,
    output_path,
    probability_csv_path=None,
    min_region_pixels=500,
    max_regions=30,
    pixel_logits=None,
):
    '''
    Salva una figura con immagine originale e classi predette dal modello.

    La figura contiene immagine originale, segmentazione colorata e confidence map.
    Se `probability_csv_path` non è `None`, viene salvato anche un CSV con le
    probabilità medie delle 19 classi per ogni regione predetta sufficientemente
    grande.
    '''
    # Se i logits non sono già disponibili, eseguo l'inferenza completa sul modello.
    if pixel_logits is None:
        original_image, pixel_logits = compute_pixel_logits_for_image(image_path, model, device)
    else:
        original_image = Image.open(image_path).convert("RGB")

    # Calcolo predizione semantica, probabilità complete e confidence per pixel.
    prediction, probabilities, confidence = compute_semantic_prediction_and_probabilities(pixel_logits)

    # Converto la maschera predetta in una immagine RGB con palette Cityscapes.
    colored_prediction = colorize_semantic_prediction(prediction)

    # Recupero le classi effettivamente presenti, così la legenda resta compatta.
    present_classes = np.unique(prediction)

    # Costruisco gli elementi della legenda usando gli stessi colori della maschera.
    legend_handles = [
        Patch(
            facecolor=CITYSCAPES_PALETTE[class_id] / 255.0,
            edgecolor="black",
            label=f"{class_id}: {CITYSCAPES_CLASSES[class_id]}",
        )
        for class_id in present_classes
    ]

    # Preparo l'immagine originale nella stessa dimensione della predizione.
    original_np = resize_original_for_plot(original_image, size=(prediction.shape[1], prediction.shape[0]))

    # Creo una figura a tre pannelli: input, predizione colorata, confidence.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Primo pannello: immagine originale.
    axes[0].imshow(original_np)
    axes[0].set_title("Immagine originale")

    # Secondo pannello: classi predette colorate.
    axes[1].imshow(colored_prediction)
    axes[1].set_title("Classi predette EoMT")
    axes[1].legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=3,
        fontsize=7,
        frameon=False,
    )

    # Terzo pannello: probabilità della classe vincente, pixel per pixel.
    conf_plot = axes[2].imshow(confidence, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[2].set_title("Confidence classe predetta")
    fig.colorbar(conf_plot, ax=axes[2], fraction=0.046, pad=0.04)

    # Rimuovo gli assi per avere un output più pulito.
    for ax in axes:
        ax.axis("off")

    # Creo la cartella di output se non esiste.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Salvo la visualizzazione su disco.
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    # Se richiesto, salvo anche il CSV con probabilità medie per regione.
    if probability_csv_path is not None:
        region_rows = summarize_predicted_regions(
            prediction=prediction,
            probabilities=probabilities,
            min_region_pixels=min_region_pixels,
            max_regions=max_regions,
        )
        save_region_probability_csv(region_rows, probability_csv_path)

    # Ritorno i prodotti intermedi per eventuale uso programmatico.
    return prediction, probabilities, confidence, pixel_logits


def build_output_paths(image_path, output_dir):
    '''
    Costruisce i path di output standard per una immagine di input.

    Per ogni immagine vengono creati nomi stabili basati sullo stem del file:
    uno per RbA, uno per la predizione semantica e uno per il CSV delle probabilità.
    '''
    # Uso lo stem del file per creare nomi leggibili e indipendenti dall'estensione.
    image_stem = Path(image_path).stem

    # Creo la cartella di output come `Path` per comporre i nomi in modo robusto.
    output_dir = Path(output_dir)

    # Ritorno tutti i path necessari allo script.
    return {
        "rba": output_dir / f"{image_stem}_rba.png",
        "prediction": output_dir / f"{image_stem}_prediction.png",
        "probabilities": output_dir / f"{image_stem}_predicted_regions_probabilities.csv",
    }


def visualize_single_image(
    image_path,
    model,
    device,
    output_dir="visualizations",
    mode="both",
    min_region_pixels=500,
    max_regions=30,
):
    '''
    Visualizza una singola immagine con RbA, predizione semantica o entrambe.

    La funzione esegue una sola inferenza EoMT per immagine e riusa gli stessi
    logits per tutte le visualizzazioni richieste, così il codice resta coerente
    e non spreca memoria/tempo.
    '''
    # Creo i path di output associati all'immagine.
    output_paths = build_output_paths(image_path, output_dir)

    # Eseguo una sola inferenza completa, inclusa la parte dei crop.
    _, pixel_logits = compute_pixel_logits_for_image(image_path, model, device)

    # Salvo la visualizzazione RbA quando richiesta.
    if mode in ("rba", "both"):
        save_rba_anomaly_visualization(
            image_path=image_path,
            model=model,
            device=device,
            output_path=output_paths["rba"],
            pixel_logits=pixel_logits,
        )

    # Salvo la visualizzazione semantica quando richiesta.
    if mode in ("prediction", "both"):
        save_semantic_prediction_visualization(
            image_path=image_path,
            model=model,
            device=device,
            output_path=output_paths["prediction"],
            probability_csv_path=output_paths["probabilities"],
            min_region_pixels=min_region_pixels,
            max_regions=max_regions,
            pixel_logits=pixel_logits,
        )

    # Calcolo le quattro mappe anomaly da usare per le metriche finali.
    anomaly_score_maps = compute_anomaly_score_maps(pixel_logits)

    # Libero memoria GPU dopo aver finito con l'immagine corrente.
    del pixel_logits
    if device == "cuda":
        torch.cuda.empty_cache()

    # Ritorno path e anomaly score, cosi il main puo stampare file e metriche aggregate.
    return output_paths, anomaly_score_maps


def main():
    '''
    Entry point da riga di comando per visualizzare immagini con EoMT.

    Esempio:
    `python evalAnomalyEoMTVisualize.py --input "datasets/.../images/*.jpg" --mode both`
    '''
    # Definisco gli argomenti CLI mantenendo i default coerenti con `evalAnomalyEoMT.py`.
    parser = ArgumentParser()
    parser.add_argument("--input", required=True, help="Path o glob delle immagini da visualizzare.")
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
        choices=["rba", "prediction", "both"],
        default="both",
        help="Tipo di visualizzazione da salvare.",
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
        help="Device da usare. Se omesso, usa CUDA quando disponibile.",
    )
    args = parser.parse_args()

    # Carico una sola volta il modello e i pesi `.bin`.
    model, device = load_eomt_for_visualization(
        config_path=args.config_path,
        state_dict_path=args.state_dict_path,
        device=args.device,
    )

    # Espando il path input come glob, esattamente nello spirito degli script eval.
    image_paths = sorted(glob.glob(os.path.expanduser(str(args.input))))

    # Interrompo con un messaggio chiaro se il glob non trova immagini.
    if not image_paths:
        raise FileNotFoundError(f"Nessuna immagine trovata con input: {args.input}")

    # Preparo gli accumulatori per calcolare le metriche anomaly alla fine del ciclo.
    metric_storage = create_empty_metric_storage()

    # Processo una immagine alla volta per contenere la memoria GPU/CPU.
    for image_path in image_paths:
        print(f"Visualizzo: {image_path}")
        output_paths, anomaly_score_maps = visualize_single_image(
            image_path=image_path,
            model=model,
            device=device,
            output_dir=args.output_dir,
            mode=args.mode,
            min_region_pixels=args.min_region_pixels,
            max_regions=args.max_regions,
        )

        # Accumulo ground truth e score per calcolare AUPRC/FPR@TPR95 a fine ciclo.
        add_image_to_metric_storage(
            image_path=image_path,
            anomaly_score_maps=anomaly_score_maps,
            metric_storage=metric_storage,
        )

        # Stampo i file creati, così è facile trovarli da terminale.
        if args.mode in ("rba", "both"):
            print(f"  RbA salvato in: {output_paths['rba']}")
        if args.mode in ("prediction", "both"):
            print(f"  Predizione salvata in: {output_paths['prediction']}")
            print(f"  Probabilità regioni salvate in: {output_paths['probabilities']}")


    # A fine ciclo stampo le metriche aggregate, oppure quelle della singola immagine.
    print_anomaly_metric_results(metric_storage)


if __name__ == "__main__":
    main()
