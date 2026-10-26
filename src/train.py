"""
train.py
Treinamento centralizado da ResNet-50 no PCOSGen-train.
Experimento 1 da metodologia: treino + validacao interna.

Salva numa pasta propria do experimento (--out-dir; padrao
outputs/experimentos/luan_pr1_AAAAMMDD_HHMMSS, com a hora do inicio):
- Melhor modelo em <out-dir>/models/best_model.pt
- Log por epoca em <out-dir>/logs/train_log.csv
- Configuracao em <out-dir>/config.json
- Particao treino/validacao em <out-dir>/splits/ (conferida contra
  resultados/splits/, que e' a do experimento de referencia)

Nunca grava em outputs/models/ nem outputs/logs/: la ficam o checkpoint e o
log do experimento de referencia (versao 3 abaixo, treinado em 2026-06-20).

Versao 3 (experimento de referencia):
- lr=1e-4 (mantido da v2)
- ReduceLROnPlateau scheduler (mantido da v2)
- weight_decay aumentado para 1e-3 (mais regularizacao)
- Dropout(0.5) na camada final do modelo
- augmentation simplificada (sem ColorJitter)

Versao 4 (PR #1 do Luan):
- treino em duas fases: 3 epocas so' com a cabeca (backbone congelado,
  lr=1e-3), depois fine-tuning completo com lr=1e-4 e otimizador/scheduler novos
- seeds de random/numpy/torch/cuda e cuDNN deterministico
- checkpoint tambem guarda otimizador, scheduler, lr e seed
- augmentation com ColorJitter leve e RandomAffine (ver transforms_config.py)
"""

import argparse
import json
import subprocess
import time
import csv
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from dataset import (
    PCOSGenTrainDataset,
    load_pcosgen_train_labels,
    split_train_val,
)
from transforms_config import get_train_transform, get_eval_transform
from model import build_resnet50

# --- configuracao ---
LABELS_PATH = Path("G:/tcc/data/pcosgen/train/PCOSGen-train/PCOSGen-train/class_label.xlsx")
IMG_DIR = Path("G:/tcc/data/pcosgen/train/PCOSGen-train/PCOSGen-train/images")
OUTPUT_DIR = Path("G:/tcc/outputs")
EXPERIMENTS_DIR = OUTPUT_DIR / "experimentos"
# particao usada pelo experimento de referencia (exportada por exportar_splits.py)
REFERENCE_SPLITS_DIR = Path("G:/tcc/resultados/splits")

# hiperparametros
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 50
HEAD_LR = 1e-3
FINE_TUNE_LR = 1e-4
WEIGHT_DECAY = 1e-3
EARLY_STOPPING_PATIENCE = 15
HEAD_ONLY_EPOCHS = 3
NUM_WORKERS = 4

# reprodutibilidade
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(SEED)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_logits = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) > 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_logits.extend(logits.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    auc = roc_auc_score(all_labels, all_logits)
    f1 = f1_score(all_labels, all_preds)
    return avg_loss, acc, auc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_logits = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) > 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_logits.extend(logits.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    auc = roc_auc_score(all_labels, all_logits)
    f1 = f1_score(all_labels, all_preds)
    return avg_loss, acc, auc, f1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="pasta do experimento (padrao: outputs/experimentos/luan_pr1_<data_hora>)",
    )
    return parser.parse_args()


def git_commit():
    try:
        return subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def check_same_split(train_df, val_df):
    """Confere se a particao e' identica a do experimento de referencia."""
    import pandas as pd
    for name, part in [("treino", train_df), ("validacao", val_df)]:
        ref = pd.read_csv(REFERENCE_SPLITS_DIR / f"pcosgen_train_{name}.csv")
        same = (
            len(ref) == len(part)
            and (ref["imagePath"].astype(str).values == part["imagePath"].astype(str).values).all()
            and (ref["label"].values == part["label"].values).all()
        )
        if not same:
            raise SystemExit(
                f"Particao de {name} difere de {REFERENCE_SPLITS_DIR}. "
                "Abortando para nao comparar experimentos com dados diferentes."
            )
    print(f"Particao identica a do experimento de referencia ({REFERENCE_SPLITS_DIR})")


def main():
    args = parse_args()
    started = datetime.now()
    exp_dir = args.out_dir or EXPERIMENTS_DIR / f"luan_pr1_{started:%Y%m%d_%H%M%S}"
    exp_dir = exp_dir.resolve()
    if exp_dir in {OUTPUT_DIR.resolve(), (OUTPUT_DIR / "models").resolve(), (OUTPUT_DIR / "logs").resolve()}:
        raise SystemExit(f"{exp_dir} guarda o experimento de referencia; use outra pasta.")
    MODEL_PATH = exp_dir / "models" / "best_model.pt"
    LOG_PATH = exp_dir / "logs" / "train_log.csv"
    if MODEL_PATH.exists() or LOG_PATH.exists():
        raise SystemExit(f"{exp_dir} ja tem checkpoint ou log; escolha outra pasta.")
    print(f"Pasta do experimento: {exp_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. carregar rotulos e dividir treino/val ---
    print("\nCarregando rotulos...")
    df = load_pcosgen_train_labels(LABELS_PATH)
    train_df, val_df = split_train_val(df, val_size=0.2, seed=SEED)
    print(f"Treino: {len(train_df)} | Validacao: {len(val_df)}")
    print(f"Distribuicao treino : {train_df['label'].value_counts().to_dict()}")
    print(f"Distribuicao validac: {val_df['label'].value_counts().to_dict()}")
    check_same_split(train_df, val_df)
    (exp_dir / "splits").mkdir(exist_ok=True)
    train_df.to_csv(exp_dir / "splits" / "pcosgen_train_treino.csv", index=False)
    val_df.to_csv(exp_dir / "splits" / "pcosgen_train_validacao.csv", index=False)

    config = {
        "experimento": exp_dir.name,
        "inicio": started.isoformat(timespec="seconds"),
        "origem": "PR #1 (Luan), commit 9cb9922: fix: train parameters and logging",
        "git_commit_base": git_commit(),
        "labels_path": str(LABELS_PATH),
        "img_dir": str(IMG_DIR),
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "num_epochs_max": NUM_EPOCHS,
        "head_only_epochs": HEAD_ONLY_EPOCHS,
        "head_lr": HEAD_LR,
        "fine_tune_lr": FINE_TUNE_LR,
        "weight_decay": WEIGHT_DECAY,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "scheduler": "ReduceLROnPlateau(mode=max sobre val_auc, factor=0.5, patience=5, min_lr=1e-6)",
        "criterio_melhor_checkpoint": "maior val_auc (estritamente maior)",
        "limiar": 0.5,
        "train_transform": repr(get_train_transform()),
        "eval_transform": repr(get_eval_transform()),
        "device": str(device),
        "torch": torch.__version__,
        "model_path": str(MODEL_PATH),
        "log_path": str(LOG_PATH),
    }
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # --- 2. class weights ---
    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"].values,
    )
    pos_weight = torch.tensor(weights[1] / weights[0]).to(device)
    print(f"\nClass weights: normal={weights[0]:.3f}, PCOS={weights[1]:.3f}")
    print(f"pos_weight para BCE: {pos_weight.item():.3f}")

    # --- 3. datasets e dataloaders ---
    train_ds = PCOSGenTrainDataset(IMG_DIR, train_df, transform=get_train_transform())
    val_ds = PCOSGenTrainDataset(IMG_DIR, val_df, transform=get_eval_transform())

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    print(f"\nBatches treino: {len(train_loader)} | Batches val: {len(val_loader)}")

    # --- 4. modelo, loss, otimizador, scheduler ---
    model = build_resnet50(pretrained=True, freeze_backbone=True).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=HEAD_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    # --- 5. loop de treinamento ---
    print(f"\nIniciando treinamento por ate {NUM_EPOCHS} epocas...")
    print(f"Early stopping com paciencia de {EARLY_STOPPING_PATIENCE} epocas")
    print(f"Learning rate inicial: {HEAD_LR}")
    print(f"Fine-tuning {HEAD_ONLY_EPOCHS + 1} com lr={FINE_TUNE_LR}")
    print(f"Weight decay: {WEIGHT_DECAY}\n")

    best_val_auc = 0.0
    patience_counter = 0
    log_rows = []

    for epoch in range(1, NUM_EPOCHS + 1):
        if epoch == HEAD_ONLY_EPOCHS + 1:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=FINE_TUNE_LR,
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            )

        t0 = time.time()

        train_loss, train_acc, train_auc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_auc, val_f1 = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_auc": train_auc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "val_f1": val_f1,
            "lr": current_lr,
            "time_sec": elapsed,
        })

        print(
            f"Ep {epoch:3d} | "
            f"trn loss={train_loss:.4f} acc={train_acc:.3f} auc={train_auc:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f} auc={val_auc:.3f} f1={val_f1:.3f} | "
            f"lr={current_lr:.1e} | "
            f"{elapsed:.1f}s"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_auc": val_auc,
                "val_f1": val_f1,
                "lr": current_lr,
                "seed": SEED,
            }, MODEL_PATH)
            print(f"  -> melhor modelo salvo (AUC={val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping: {EARLY_STOPPING_PATIENCE} epocas sem melhora")
                break

    # --- 6. salva log ---
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    config["fim"] = datetime.now().isoformat(timespec="seconds")
    config["epocas_executadas"] = len(log_rows)
    config["melhor_epoca"] = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)["epoch"]
    config["melhor_val_auc"] = best_val_auc
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\nTreinamento concluido.")
    print(f"Melhor AUC de validacao: {best_val_auc:.4f}")
    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Log salvo em: {LOG_PATH}")


if __name__ == "__main__":
    main()