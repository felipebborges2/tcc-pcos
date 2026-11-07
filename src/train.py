"""
train.py
Treinamento centralizado da ResNet-50 no PCOSGen-train.
Experimento 1 da metodologia: treino + validacao interna.

Salva numa pasta propria do experimento (--out-dir; padrao
outputs/experimentos/treino_AAAAMMDD_HHMMSS, com a hora do inicio):
- Melhor modelo em <out-dir>/models/best_model.pt
- Log por epoca em <out-dir>/logs/train_log.csv (gravado a cada epoca)
- Configuracao em <out-dir>/config.json
- Particao treino/validacao em <out-dir>/splits/

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

Versao 5 (relatorio tecnico do Luan). Sem opcoes, o treino e' o da versao 4.
Cada mudanca do relatorio e' uma opcao, para a ablacao passo a passo:
- sempre: limiar de Youden calculado na validacao do melhor checkpoint e salvo
  nele; log gravado a cada epoca; worker_init_fn e generator nos DataLoaders
- --split grupo: divisao treino/validacao por grupo de duplicatas (dedup.py)
- --preproc gray_pad: escala de cinza + PadToSquare
- --augment ultrassom: augmentation para ultrassom (exige gray_pad)
- --otimizacao adamw_gradual: AdamW (weight_decay=0.05), LR 1e-5 no backbone e
  1e-3 na cabeca, descongelamento gradual por estagio; scheduler, early
  stopping e escolha do melhor checkpoint pela val_loss
- --label-smoothing 0.05: label smoothing na loss de treino
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
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.utils.class_weight import compute_class_weight

from dataset import (
    PCOSGenTrainDataset,
    load_pcosgen_train_labels,
    split_train_val,
    split_train_val_grupos,
)
from transforms_config import get_train_transform, get_eval_transform, PREPROCS, AUGMENTS
from model import build_resnet50, set_trainable_stages, get_param_groups

# --- configuracao ---
LABELS_PATH = Path("G:/tcc/data/pcosgen/train/PCOSGen-train/PCOSGen-train/class_label.xlsx")
IMG_DIR = Path("G:/tcc/data/pcosgen/train/PCOSGen-train/PCOSGen-train/images")
OUTPUT_DIR = Path("G:/tcc/outputs")
EXPERIMENTS_DIR = OUTPUT_DIR / "experimentos"
# particao usada pelo experimento de referencia (exportada por exportar_splits.py)
REFERENCE_SPLITS_DIR = Path("G:/tcc/resultados/splits")
GROUPS_PATH = OUTPUT_DIR / "dedup" / "pcosgen_train_groups.csv"

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

# --otimizacao adamw_gradual (relatorio, secao 4.3)
LR_BACKBONE = 1e-5
LR_HEAD = 1e-3
ADAMW_WEIGHT_DECAY = 0.05
UNFREEZE_SCHEDULE = {1: 0, 4: 1, 7: 2, 10: 4}  # epoca -> n de estagios liberados


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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def smooth(labels, eps):
    """Label smoothing para BCE: 0 -> eps/2, 1 -> 1 - eps/2."""
    return labels * (1 - eps) + 0.5 * eps


def youden_threshold(labels, logits):
    """Limiar de probabilidade que maximiza sensibilidade + especificidade - 1.

    Calculado com a mesma sigmoide em float32 de evaluate.py. Como evaluate.py
    classifica PCOS com prob > limiar, devolve o maior float32 abaixo do ponto
    escolhido, o que reproduz prob >= ponto (convencao do roc_curve).
    """
    probs = torch.sigmoid(torch.as_tensor(np.asarray(logits, dtype=np.float32))).numpy()
    fpr, tpr, thr = roc_curve(labels, probs)
    t = np.float32(thr[np.argmax(tpr - fpr)])
    return float(np.nextafter(t, np.float32(-np.inf), dtype=np.float32))


def train_one_epoch(model, loader, criterion, optimizer, device, label_smoothing=0.0):
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
        target = smooth(labels, label_smoothing) if label_smoothing > 0 else labels
        loss = criterion(logits, target)
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
    """Loss com os rotulos originais (sem label smoothing)."""
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
    return avg_loss, acc, auc, f1, np.array(all_labels), np.array(all_logits)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="pasta do experimento (padrao: outputs/experimentos/treino_<data_hora>)",
    )
    parser.add_argument("--split", choices=["aleatorio", "grupo"], default="aleatorio")
    parser.add_argument("--grupos", type=Path, default=GROUPS_PATH,
                        help="pcosgen_train_groups.csv gerado por dedup.py (para --split grupo)")
    parser.add_argument("--preproc", choices=PREPROCS, default="rgb_resize")
    parser.add_argument("--augment", choices=AUGMENTS, default="pr1")
    parser.add_argument("--otimizacao", choices=["pr1", "adamw_gradual"], default="pr1")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--epocas", type=int, default=NUM_EPOCHS)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--debug-amostras", type=int, default=0,
                        help="so' para teste rapido: usa N imagens por classe em treino e validacao")
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


def make_loader(ds, shuffle, num_workers, device):
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=g,
    )


def main():
    args = parse_args()
    if args.augment == "ultrassom" and args.preproc != "gray_pad":
        raise SystemExit("--augment ultrassom exige --preproc gray_pad")
    adamw = args.otimizacao == "adamw_gradual"
    started = datetime.now()
    exp_dir = args.out_dir or EXPERIMENTS_DIR / f"treino_{started:%Y%m%d_%H%M%S}"
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
    if args.split == "grupo":
        grupos = pd.read_csv(args.grupos)
        if not (grupos["imagePath"].astype(str).values == df["imagePath"].astype(str).values).all():
            raise SystemExit(f"{args.grupos} nao corresponde ao arquivo de rotulos do treino.")
        train_df, val_df = split_train_val_grupos(df, grupos["group_id"].values, seed=SEED)
        mistos = set(train_df["imagePath"]) & set(val_df["imagePath"])
        assert not mistos
        print(f"Divisao por grupo de duplicatas ({args.grupos})")
    else:
        train_df, val_df = split_train_val(df, val_size=0.2, seed=SEED)
        if not args.debug_amostras:
            check_same_split(train_df, val_df)
    if args.debug_amostras:
        n = args.debug_amostras
        train_df = train_df.groupby("label").head(n).reset_index(drop=True)
        val_df = val_df.groupby("label").head(n).reset_index(drop=True)
        print(f"[DEBUG] usando {n} imagens por classe")
    print(f"Treino: {len(train_df)} | Validacao: {len(val_df)}")
    print(f"Distribuicao treino : {train_df['label'].value_counts().to_dict()}")
    print(f"Distribuicao validac: {val_df['label'].value_counts().to_dict()}")
    (exp_dir / "splits").mkdir(exist_ok=True)
    train_df.to_csv(exp_dir / "splits" / "pcosgen_train_treino.csv", index=False)
    val_df.to_csv(exp_dir / "splits" / "pcosgen_train_validacao.csv", index=False)

    train_transform = get_train_transform(args.preproc, args.augment)
    eval_transform = get_eval_transform(args.preproc)
    criterio = "menor val_loss" if adamw else "maior val_auc"
    config = {
        "experimento": exp_dir.name,
        "inicio": started.isoformat(timespec="seconds"),
        "git_commit_base": git_commit(),
        "labels_path": str(LABELS_PATH),
        "img_dir": str(IMG_DIR),
        "split": args.split,
        "grupos": str(args.grupos) if args.split == "grupo" else None,
        "preproc": args.preproc,
        "augment": args.augment,
        "otimizacao": args.otimizacao,
        "label_smoothing": args.label_smoothing,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "num_epochs_max": args.epocas,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "criterio_melhor_checkpoint": f"{criterio} (estritamente melhor)",
        "limiar": "Youden na validacao, no melhor checkpoint",
        "train_transform": repr(train_transform),
        "eval_transform": repr(eval_transform),
        "num_workers": args.num_workers,
        "debug_amostras": args.debug_amostras,
        "device": str(device),
        "torch": torch.__version__,
        "model_path": str(MODEL_PATH),
        "log_path": str(LOG_PATH),
    }
    if adamw:
        config.update({
            "otimizador": f"AdamW(weight_decay={ADAMW_WEIGHT_DECAY})",
            "lr_backbone": LR_BACKBONE, "lr_head": LR_HEAD,
            "unfreeze_schedule": {str(k): v for k, v in UNFREEZE_SCHEDULE.items()},
            "scheduler": "ReduceLROnPlateau(mode=min sobre val_loss, factor=0.5, patience=3, min_lr=1e-7)",
        })
    else:
        config.update({
            "otimizador": f"Adam(weight_decay={WEIGHT_DECAY})",
            "head_only_epochs": HEAD_ONLY_EPOCHS, "head_lr": HEAD_LR, "fine_tune_lr": FINE_TUNE_LR,
            "scheduler": "ReduceLROnPlateau(mode=max sobre val_auc, factor=0.5, patience=5, min_lr=1e-6)",
        })
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
    train_ds = PCOSGenTrainDataset(IMG_DIR, train_df, transform=train_transform)
    val_ds = PCOSGenTrainDataset(IMG_DIR, val_df, transform=eval_transform)
    train_loader = make_loader(train_ds, True, args.num_workers, device)
    val_loader = make_loader(val_ds, False, args.num_workers, device)
    print(f"\nBatches treino: {len(train_loader)} | Batches val: {len(val_loader)}")

    # --- 4. modelo, loss, otimizador, scheduler ---
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if adamw:
        model = build_resnet50(pretrained=True, freeze_backbone=False).to(device)
        # todos os parametros desde o inicio; os congelados ficam sem grad e o AdamW os ignora
        optimizer = torch.optim.AdamW(
            get_param_groups(model, LR_BACKBONE, LR_HEAD),
            weight_decay=ADAMW_WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7,
        )
    else:
        model = build_resnet50(pretrained=True, freeze_backbone=True).to(device)
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
    print(f"\nIniciando treinamento por ate {args.epocas} epocas...")
    print(f"Early stopping com paciencia de {EARLY_STOPPING_PATIENCE} epocas ({criterio})")
    if adamw:
        print(f"AdamW: lr backbone={LR_BACKBONE}, lr cabeca={LR_HEAD}, weight decay={ADAMW_WEIGHT_DECAY}")
        print(f"Descongelamento gradual (epoca -> estagios): {UNFREEZE_SCHEDULE}")
    else:
        print(f"Learning rate inicial: {HEAD_LR}")
        print(f"Fine-tuning {HEAD_ONLY_EPOCHS + 1} com lr={FINE_TUNE_LR}")
        print(f"Weight decay: {WEIGHT_DECAY}")
    if args.label_smoothing > 0:
        print(f"Label smoothing: {args.label_smoothing}")
    print()

    best_score = None
    best_val_auc = None
    patience_counter = 0
    n_epochs = 0

    for epoch in range(1, args.epocas + 1):
        if adamw:
            if epoch in UNFREEZE_SCHEDULE:
                set_trainable_stages(model, UNFREEZE_SCHEDULE[epoch])
        elif epoch == HEAD_ONLY_EPOCHS + 1:
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
            model, train_loader, criterion, optimizer, device, args.label_smoothing
        )
        val_loss, val_acc, val_auc, val_f1, val_labels, val_logits = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss if adamw else val_auc)
        current_lr = optimizer.param_groups[0]["lr"]
        head_lr = optimizer.param_groups[-1]["lr"]
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        elapsed = time.time() - t0
        n_epochs = epoch

        row = {
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
            "lr_head": head_lr,
            "trainable_params": trainable,
            "time_sec": elapsed,
        }
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)

        print(
            f"Ep {epoch:3d} | "
            f"trn loss={train_loss:.4f} acc={train_acc:.3f} auc={train_auc:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f} auc={val_auc:.3f} f1={val_f1:.3f} | "
            f"lr={current_lr:.1e} | "
            f"{elapsed:.1f}s"
        )

        score = -val_loss if adamw else val_auc
        if best_score is None or score > best_score:
            best_score = score
            best_val_auc = val_auc
            patience_counter = 0
            threshold = youden_threshold(val_labels, val_logits)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_auc": val_auc,
                "val_f1": val_f1,
                "val_loss": val_loss,
                "lr": current_lr,
                "seed": SEED,
                "threshold": threshold,
                "threshold_origem": "Youden na validacao",
                "preproc": args.preproc,
            }, MODEL_PATH)
            print(f"  -> melhor modelo salvo ({criterio}: AUC={val_auc:.4f}, loss={val_loss:.4f}, "
                  f"limiar Youden={threshold:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping: {EARLY_STOPPING_PATIENCE} epocas sem melhora")
                break

    best = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    config["fim"] = datetime.now().isoformat(timespec="seconds")
    config["epocas_executadas"] = n_epochs
    config["melhor_epoca"] = best["epoch"]
    config["melhor_val_auc"] = best["val_auc"]
    config["melhor_val_loss"] = best["val_loss"]
    config["limiar_youden"] = best["threshold"]
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\nTreinamento concluido.")
    print(f"Melhor epoca: {best['epoch']} (AUC val={best['val_auc']:.4f}, loss val={best['val_loss']:.4f})")
    print(f"Limiar de Youden (validacao): {best['threshold']:.4f}")
    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Log salvo em: {LOG_PATH}")


if __name__ == "__main__":
    main()
