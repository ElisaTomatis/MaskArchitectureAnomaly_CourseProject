import torch
import random
import numpy as np
import yaml
import warnings
import torch.nn.functional as F

from argparse import ArgumentParser

from functions import *
from eomt.datasets.cityscapes_semantic import CityscapesSemanticOE

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def main():
    parser = ArgumentParser()
    parser.add_argument("--cityscapes-path",
        type=str,
        required=True,
        help="/content/cityscapes/",
    )
    parser.add_argument(
        "--coco-root",
        type=str,
        required=True,
        help="/content/drive/MyDrive/cityscapes/coco",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="/content/drive/MyDrive/eomt_cityscapes_oe_finetuned.pth",
    ) # dove mettere i pesi aggiornati dopo finetuning
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--p-ood", type=float, default=0.5)
    parser.add_argument("--lambda-oe", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    use_cuda = (not args.cpu) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    results_path = '/content/drive/MyDrive/results_finetune.txt'
    print("Scrivo risultati in:", results_path)
    file = open(results_path, 'w')
    file.flush()
    
    config_path = '../configs/dinov2/cityscapes/semantic/eomt_base_640.yaml'
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    state_dict_path = '/content/drive/MyDrive/eomt_cityscapes.bin'
    warnings.filterwarnings("ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )
    
    # carica il modello
    model = setup_model(config, state_dict_path, device)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # l'optimizer prende i parametri solo non frizzati

    print("Preparing dataset...")
    data_module = CityscapesSemanticOE(
        path=args.cityscapes_path,
        coco_root=args.coco_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        p_ood=args.p_ood,
        img_size=(1024, 1024),
        check_empty_targets=True,
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()

    print("Starting OE fine-tuning...")

    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_oe=args.lambda_oe,
            margin=args.margin,
            file=file,
        )

        msg = (
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"loss={avg_loss['loss']:.6f} | "
            f"loss_seg={avg_loss['loss_seg']:.6f} | "
            f"loss_ood={avg_loss['loss_ood']:.6f}"
        )

        print(msg)

        file.write(msg + "\n")
        file.flush()
        
        torch.save(
            model.state_dict(),
            args.save_path,
        )

        print(f"Checkpoint saved to: {args.save_path}")

    print("Training completed.")
    file.close()


if __name__ == "__main__":
    main()