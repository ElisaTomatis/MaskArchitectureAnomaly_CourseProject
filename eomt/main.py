# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from PyTorch Lightning,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------



import jsonargparse._typehints as _t
import os
from types import MethodType
from pathlib import Path
from gitignore_parser import parse_gitignore
import logging
import torch
import warnings
from lightning.pytorch import cli
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loops.training_epoch_loop import _TrainingEpochLoop
from lightning.pytorch.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher

from training.lightning_module import LightningModule
from datasets.lightning_data_module import LightningDataModule

# Suppress PyTorch FX warnings for DINOv3 models
import os
os.environ["TORCH_LOGS"] = "-dynamo"


def _default_run_root() -> Path:
    """
    Determina la cartella radice in cui salvare output e checkpoint.

    Se il codice viene eseguito in Google Colab e il Drive risulta montato,
    utilizza una directory dedicata all'interno di Drive. In caso contrario
    usa una cartella locale nella directory corrente.

    Returns:
        Path della directory radice per run e checkpoint.
    """

    colab_drive = Path("/content/drive/MyDrive")
    if colab_drive.exists():
        return colab_drive / "eomt_runs"
    return Path.cwd() / "eomt_runs"


def _find_model_checkpoint_callback(trainer) -> ModelCheckpoint | None:
    """
    Cerca il callback ModelCheckpoint registrato nel trainer.

    Args:
        trainer: Istanza del trainer Lightning.

    Returns:
        Il callback ModelCheckpoint se presente, altrimenti None.
    """

    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            return callback
    return None


def _format_hparam_for_filename(value) -> str:
    """
    Converte un iperparametro in una stringa sicura da usare nei nomi file.

    I caratteri che possono creare problemi nei nomi dei checkpoint vengono
    sostituiti con rappresentazioni compatibili.

    Args:
        value: Valore dell'iperparametro.

    Returns:
        Stringa utilizzabile all'interno di un nome file.
    """

    if isinstance(value, float):
        value = f"{value:.6g}"
    return str(value).replace("-", "m").replace("+", "").replace(".", "p")


_orig_single = _t.raise_unexpected_value


def _raise_single(*args, exception=None, **kwargs):
    """
    Propaga l'eccezione originale prodotta da jsonargparse.

    Questa patch evita che alcuni errori di validazione dei tipi vengano
    sostituiti da messaggi meno informativi.

    Args:
        *args: Argomenti inoltrati alla funzione originale.
        exception: Eccezione eventualmente prodotta dal parser.
        **kwargs: Argomenti keyword inoltrati alla funzione originale.

    Returns:
        Risultato della funzione originale quando non viene sollevata
        alcuna eccezione.
    """
    if isinstance(exception, Exception):
        raise exception
    return _orig_single(*args, exception=exception, **kwargs)


_orig_union = _t.raise_union_unexpected_value


def _raise_union(subtypes, val, vals):
    """
    Propaga l'errore più specifico durante la validazione di tipi union.

    Quando jsonargparse prova più sottotipi, questa funzione restituisce
    l'eccezione più informativa disponibile.

    Args:
        subtypes: Tipi ammessi dalla union.
        val: Valore da validare.
        vals: Risultati o eccezioni dei tentativi di validazione.

    Returns:
        Risultato della funzione originale se non sono presenti eccezioni.
    """
    for e in reversed(vals):
        if isinstance(e, Exception):
            raise e
    return _orig_union(subtypes, val, vals)


_t.raise_unexpected_value = _raise_single
_t.raise_union_unexpected_value = _raise_union


def _should_check_val_fx(self: _TrainingEpochLoop, data_fetcher: _DataFetcher) -> bool:
    """
    Determina se eseguire la validazione durante il training.

    Estende la logica standard di Lightning supportando controlli di
    validazione basati sul global step quando vengono usati gradient
    accumulation e validazione a intervalli di iterazioni.

    Args:
        self: Training epoch loop di Lightning.
        data_fetcher: Data fetcher associato al loop.

    Returns:
        True se deve essere eseguita la validazione, False altrimenti.
    """
    if not self._should_check_val_epoch():
        return False

    is_infinite_dataset = self.trainer.val_check_batch == float("inf")
    is_last_batch = self.batch_progress.is_last_batch
    if is_last_batch and (
        is_infinite_dataset or isinstance(data_fetcher, _DataLoaderIterDataFetcher)
    ):
        return True

    if self.trainer.should_stop and self.trainer.fit_loop._can_stop_early:
        return True

    is_val_check_batch = is_last_batch
    if isinstance(self.trainer.limit_train_batches, int) and is_infinite_dataset:
        is_val_check_batch = (
            self.batch_idx + 1
        ) % self.trainer.limit_train_batches == 0
    elif self.trainer.val_check_batch != float("inf"):
        if self.trainer.check_val_every_n_epoch is not None:
            is_val_check_batch = (
                self.batch_idx + 1
            ) % self.trainer.val_check_batch == 0
        else:
            # added below to check val based on global steps instead of batches in case of iteration based val check and gradient accumulation
            is_val_check_batch = (
                self.global_step
            ) % self.trainer.val_check_batch == 0 and not self._should_accumulate()

    return is_val_check_batch


class LightningCLI(cli.LightningCLI):
    """
    Estensione personalizzata di LightningCLI per il training EoMT.

    Configura warning, precisione numerica, collegamenti automatici tra
    parametri del DataModule e del modello, gestione checkpoint e resume
    automatico degli esperimenti.
    """
    def __init__(self, *args, **kwargs):
        """
        Inizializza la CLI configurando ambiente, warning e opzioni Torch.

        Args:
            *args: Argomenti passati alla LightningCLI base.
            **kwargs: Argomenti keyword passati alla LightningCLI base.
        """
        logging.getLogger().setLevel(logging.INFO)
        torch.set_float32_matmul_precision("medium")
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True
        warnings.filterwarnings(
            "ignore",
            message=r".*It is recommended to use .* when logging on epoch level in distributed setting to accumulate the metric across devices.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"^The ``compute`` method of metric PanopticQuality was called before the ``update`` method.*",
        )
        warnings.filterwarnings(
            "ignore", message=r"^Grad strides do not match bucket view strides.*"
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*functools.partial will be a method descriptor in future Python versions*",
        )

        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        """
        Registra argomenti personalizzati e collega parametri condivisi.

        I collegamenti mantengono coerenti numero di classi, dimensioni delle
        immagini, classi stuff e checkpoint tra DataModule, modello, network ed
        encoder.

        Args:
            parser: Parser utilizzato da LightningCLI.

        Returns:
            None.
        """
        parser.add_argument("--compile_disabled", action="store_true")

        parser.add_argument(
            "--scheduler",
            default=None,
            help=(
                "Scheduler LR opzionale da inoltrare al modello. "
                "Esempi: cosine_warmup, two_stage_warmup_poly, none."
            ),
        )

        parser.add_argument(
            "--resume_disabled",
            action="store_true",
            help="Disabilita il resume automatico dall'ultimo checkpoint trovato.",
        )

        parser.link_arguments(
            "data.init_args.num_classes", "model.init_args.num_classes"
        )
        parser.link_arguments(
            "data.init_args.num_classes",
            "model.init_args.network.init_args.num_classes",
        )

        parser.link_arguments(
            "data.init_args.stuff_classes", "model.init_args.stuff_classes"
        )

        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.link_arguments(
            "data.init_args.img_size", "model.init_args.network.init_args.img_size"
        )
        parser.link_arguments(
            "data.init_args.img_size",
            "model.init_args.network.init_args.encoder.init_args.img_size",
        )

        parser.link_arguments(
            "model.init_args.ckpt_path",
            "model.init_args.network.init_args.encoder.init_args.ckpt_path",
        )

    def fit(self, model, **kwargs):
        """
        Avvia il training configurando logging, checkpoint e resume automatico.

        Prima dell'addestramento registra il codice sorgente, aggiorna la
        configurazione dei checkpoint, applica la logica personalizzata di
        validazione e riprende automaticamente dall'ultimo checkpoint se
        disponibile.

        Args:
            model: Modello Lightning da addestrare.
            **kwargs: Argomenti inoltrati a trainer.fit.

        Returns:
            None.
        """
        if hasattr(self.trainer.logger.experiment, "log_code"):
            is_gitignored = parse_gitignore(".gitignore")
            include_fn = lambda path: path.endswith(".py") or path.endswith(".yaml")
            self.trainer.logger.experiment.log_code(
                ".", include_fn=include_fn, exclude_fn=is_gitignored
            )

        checkpoint_callback = _find_model_checkpoint_callback(self.trainer)
        if checkpoint_callback is not None:
            run_name = getattr(self.trainer.logger, "name", "default_run")
            checkpoint_dir = _default_run_root() / "checkpoints" / run_name
            checkpoint_callback.dirpath = str(checkpoint_dir)
            # Rendiamo ogni checkpoint riconoscibile per gli iperparametri OE
            # principali, evitando sovrascritture quando cambia il learning rate.
            lr_tag = _format_hparam_for_filename(getattr(model, "lr", "na"))
            lambda_oe_tag = _format_hparam_for_filename(
                getattr(model, "lambda_rba", "na")
            )
            checkpoint_callback.filename = (
                f"lr{lr_tag}-lambdaoe{lambda_oe_tag}-epoch{{epoch:03d}}-step{{step:06d}}"
            )
            checkpoint_callback.CHECKPOINT_NAME_LAST = (
                f"last-lr{lr_tag}-lambdaoe{lambda_oe_tag}"
            )
            checkpoint_callback.FILE_EXTENSION = ".ckpt"

        self.trainer.fit_loop.epoch_loop._should_check_val_fx = MethodType(
            _should_check_val_fx, self.trainer.fit_loop.epoch_loop
        )

        if not self.config[self.config["subcommand"]]["compile_disabled"]:
            model = torch.compile(model)

        # Se esiste un last.ckpt nella cartella del checkpoint, ripartiamo da lì.
        if not self.config[self.config["subcommand"]]["resume_disabled"]:
            checkpoint_callback = _find_model_checkpoint_callback(self.trainer)
            if checkpoint_callback is not None and checkpoint_callback.dirpath is not None:
                last_checkpoint = (
                    Path(checkpoint_callback.dirpath)
                    / f"{checkpoint_callback.CHECKPOINT_NAME_LAST}{checkpoint_callback.FILE_EXTENSION}"
                )
                if last_checkpoint.exists() and kwargs.get("ckpt_path") is None:
                    logging.info(f"Resuming from checkpoint: {last_checkpoint}")
                    kwargs["ckpt_path"] = str(last_checkpoint)

        self.trainer.fit(model, **kwargs)


def cli_main():
    """
    Crea e avvia la LightningCLI del progetto.

    Registra il LightningModule e il LightningDataModule, imposta i valori
    predefiniti del trainer e configura callback, precisione e seed per gli
    esperimenti.

    Returns:
        None.
    """
    LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=0,
        trainer_defaults={
            "precision": "16-mixed",
            "enable_model_summary": False,
            "default_root_dir": str(_default_run_root()),
            "callbacks": [
                ModelSummary(max_depth=3),
                LearningRateMonitor(logging_interval="epoch"),
                ModelCheckpoint(
                    dirpath=str(_default_run_root() / "checkpoints" / "pending_run"),
                    filename="epoch{epoch:03d}-step{step:06d}",
                    save_last=True,
                    save_top_k=-1,
                    every_n_epochs=1,
                ),
            ],
            "devices": 1,
            "gradient_clip_val": 0.01,
            "gradient_clip_algorithm": "norm",
        },
    )


if __name__ == "__main__":
    cli_main()
