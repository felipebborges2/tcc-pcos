"""
train.py
Treinamento centralizado da ResNet-50 no PCOSGen-train.
Experimento 1 da metodologia: treino + validacao interna.

Salva:
- Melhor modelo em outputs/models/best_model.pt
- Log por epoca em outputs/logs/train_log.csv

Versao 3:
- lr=1e-4 (mantido da v2)
- ReduceLROnPlateau scheduler (mantido da v2)
- weight_decay aumentado para 1e-3 (mais regularizacao)
- Dropout(0.5) na camada final do modelo
- augmentation simplificada (sem ColorJitter)
"""

import time
import csv
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
MODEL_PATH = OUTPUT_DIR / "models" / "best_model.pt"
LOG_PATH = OUTPUT_DIR / "logs" / "train_log.csv"

# hiperparametros
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
EARLY_STOPPING_PATIENCE = 15
NUM_WORKERS = 4

# reprodutibilidade
torch.manual_seed(SEED)
np.random.seed(SEED)


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- 1. carregar rotulos e dividir treino/val ---
    print("\nCarregando rotulos...")
    df = load_pcosgen_train_labels(LABELS_PATH)
    train_df, val_df = split_train_val(df, val_size=0.2, seed=SEED)
    print(f"Treino: {len(train_df)} | Validacao: {len(val_df)}")
    print(f"Distribuicao treino : {train_df['label'].value_counts().to_dict()}")
    print(f"Distribuicao validac: {val_df['label'].value_counts().to_dict()}")

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
    model = build_resnet50(pretrained=True, freeze_backbone=False).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
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
    print(f"Learning rate inicial: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}\n")

    best_val_auc = 0.0
    patience_counter = 0
    log_rows = []

    for epoch in range(1, NUM_EPOCHS + 1):
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
                "val_auc": val_auc,
                "val_f1": val_f1,
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

    print(f"\nTreinamento concluido.")
    print(f"Melhor AUC de validacao: {best_val_auc:.4f}")
    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Log salvo em: {LOG_PATH}")


if __name__ == "__main__":
    main()