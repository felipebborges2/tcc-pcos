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
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score,
    balanced_accuracy_score, confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

from dataset import (
    PCOSGenTrainDataset,
    load_pcosgen_train_labels,
    split_train_val,
    split_train_val_grupos,
    validate_train_val_split,
)
from transforms_config import get_train_transform, get_eval_transform, PREPROCS, AUGMENTS
from model import build_resnet50, set_trainable_stages, get_param_groups, freeze_batchnorm_stats
from diagnosticos_treino import (
    save_augmentation_examples, probability_diagnostics, compare_with_step6,
)

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
CONSERVATIVE_LR_HEAD = 3e-4
CONSERVATIVE_WEIGHT_DECAY = 0.01
CONSERVATIVE_SCHEDULE = {1: 0, 6: 1, 12: 2}
CONSERVATIVE_PATIENCE = 8
REPO_DIR = Path(__file__).resolve().parent.parent
STEP6_DIR = REPO_DIR / "outputs/experimentos/ablacao_20261107_142831/passo6_label_smoothing"


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


def epoch_metrics(labels, logits, preds):
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "acc": float(np.mean(np.asarray(preds) == np.asarray(labels))),
        "auc": float(roc_auc_score(labels, logits)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "specificity": float(tn / (tn + fp)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def train_one_epoch(model, loader, criterion, optimizer, device, label_smoothing=0.0,
                    freeze_bn=False):
    model.train()
    if freeze_bn:
        freeze_batchnorm_stats(model)
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
    metrics = epoch_metrics(all_labels, all_logits, all_preds)
    metrics["loss"] = avg_loss
    return metrics


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
    metrics = epoch_metrics(all_labels, all_logits, all_preds)
    metrics["loss"] = avg_loss
    return metrics, np.array(all_labels), np.array(all_logits)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="pasta do experimento (padrao: outputs/experimentos/treino_<data_hora>)",
    )
    parser.add_argument("--experimento", choices=["passo7_generalizacao"], default=None)
    parser.add_argument("--passo6", type=Path, default=STEP6_DIR,
                        help="experimento de referencia para conferir splits e comparar validacao")
    parser.add_argument("--labels-path", type=Path, default=LABELS_PATH)
    parser.add_argument("--img-dir", type=Path, default=IMG_DIR)
    parser.add_argument("--split", choices=["aleatorio", "grupo"], default=None)
    parser.add_argument("--grupos", type=Path, default=None,
                        help="pcosgen_train_groups.csv gerado por dedup.py (para --split grupo)")
    parser.add_argument("--preproc", choices=PREPROCS, default=None)
    parser.add_argument("--augment", choices=AUGMENTS, default=None)
    parser.add_argument("--otimizacao", choices=["pr1", "adamw_gradual", "adamw_gradual_conservador"],
                        default=None)
    parser.add_argument("--freeze-bn", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--epocas", type=int, default=NUM_EPOCHS)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--debug-amostras", type=int, default=0,
                        help="so' para teste rapido: usa N imagens por classe em treino e validacao")
    args = parser.parse_args(argv)
    candidate = (args.experimento == "passo7_generalizacao"
                 or args.otimizacao == "adamw_gradual_conservador")
    defaults = {
        "split": "grupo" if candidate else "aleatorio",
        "preproc": "gray_pad" if candidate else "rgb_resize",
        "augment": "ultrassom_moderado" if candidate else "pr1",
        "otimizacao": "adamw_gradual_conservador" if candidate else "pr1",
        "freeze_bn": candidate,
        "grupos": REPO_DIR / "outputs/dedup/pcosgen_train_groups.csv" if candidate else GROUPS_PATH,
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if candidate:
        required = {"split": "grupo", "preproc": "gray_pad", "freeze_bn": True}
        if args.experimento:
            required.update(augment="ultrassom_moderado",
                            otimizacao="adamw_gradual_conservador", label_smoothing=0.0)
        for key, value in required.items():
            if getattr(args, key) != value:
                parser.error(f"Passo 7 exige {key}={value}")
        if args.out_dir is None:
            args.out_dir = args.passo6.parent / "passo7_generalizacao"
    if not 0 <= args.label_smoothing < 1:
        parser.error("--label-smoothing deve estar em [0, 1)")
    if args.epocas < 1 or args.num_workers < 0 or args.debug_amostras < 0:
        parser.error("epocas >= 1; num-workers e debug-amostras >= 0")
    return args


def git_commit():
    try:
        return subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def check_same_split(train_df, val_df, reference_dir=REFERENCE_SPLITS_DIR):
    """Confere se a particao e' identica a do experimento de referencia."""
    for name, part in [("treino", train_df), ("validacao", val_df)]:
        ref = pd.read_csv(reference_dir / f"pcosgen_train_{name}.csv")
        same = (
            len(ref) == len(part)
            and (ref["imagePath"].astype(str).values == part["imagePath"].astype(str).values).all()
            and (ref["label"].values == part["label"].values).all()
        )
        if not same:
            raise SystemExit(
                f"Particao de {name} difere de {reference_dir}. "
                "Abortando para nao comparar experimentos com dados diferentes."
            )
    print(f"Particao identica a do experimento de referencia ({reference_dir})")


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
    if args.augment in ("ultrassom", "ultrassom_moderado") and args.preproc != "gray_pad":
        raise SystemExit(f"--augment {args.augment} exige --preproc gray_pad")
    conservative = args.otimizacao == "adamw_gradual_conservador"
    adamw = args.otimizacao in ("adamw_gradual", "adamw_gradual_conservador")
    lr_head = CONSERVATIVE_LR_HEAD if conservative else LR_HEAD
    weight_decay = CONSERVATIVE_WEIGHT_DECAY if conservative else ADAMW_WEIGHT_DECAY
    unfreeze_schedule = CONSERVATIVE_SCHEDULE if conservative else UNFREEZE_SCHEDULE
    early_patience = CONSERVATIVE_PATIENCE if conservative else EARLY_STOPPING_PATIENCE
    set_seed(SEED)
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
    df = load_pcosgen_train_labels(args.labels_path)
    grupos = None
    if args.split == "grupo":
        grupos = pd.read_csv(args.grupos)
        if len(grupos) != len(df) or not np.array_equal(
            grupos["imagePath"].astype(str).values, df["imagePath"].astype(str).values
        ):
            raise SystemExit(f"{args.grupos} nao corresponde ao arquivo de rotulos do treino.")
        train_df, val_df = split_train_val_grupos(df, grupos["group_id"].values, seed=SEED)
        print(f"Divisao por grupo de duplicatas ({args.grupos})")
    else:
        train_df, val_df = split_train_val(df, val_size=0.2, seed=SEED)
        if not args.debug_amostras:
            check_same_split(train_df, val_df)
    sanity = validate_train_val_split(train_df, val_df, grupos)
    print("Rotulos confirmados: 0 = normal; 1 = PCOS. Sem sobreposicao de imagens/grupos.")
    if grupos is not None:
        grouped_labels = df.assign(group_id=grupos["group_id"].values).groupby("group_id")["label"].nunique()
        sanity["mixed_label_groups"] = int((grouped_labels > 1).sum())
        if sanity["mixed_label_groups"]:
            print(f"[aviso] {sanity['mixed_label_groups']} grupos possuem rotulos mistos; "
                  "mantidos intactos, sem alterar dados ou particao.")
    if conservative:
        baseline = json.loads((args.passo6 / "config.json").read_text(encoding="utf-8"))
        if (baseline.get("split") != "grupo" or baseline.get("seed") != SEED
                or baseline.get("debug_amostras", 0) or "fim" not in baseline):
            raise SystemExit("Passo 6 deve estar concluido, com split grupo, seed 42 e dados completos.")
        check_same_split(train_df, val_df, args.passo6 / "splits")
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
        "labels_path": str(args.labels_path),
        "img_dir": str(args.img_dir),
        "split": args.split,
        "grupos": str(args.grupos) if args.split == "grupo" else None,
        "preproc": args.preproc,
        "augment": args.augment,
        "otimizacao": args.otimizacao,
        "label_smoothing": args.label_smoothing,
        "freeze_bn": args.freeze_bn,
        "bn_affine_frozen": args.freeze_bn,
        "classes": {"0": "normal", "1": "PCOS"},
        "class_distribution": {
            "treino": {"normal": int((train_df.label == 0).sum()), "pcos": int((train_df.label == 1).sum())},
            "validacao": {"normal": int((val_df.label == 0).sum()), "pcos": int((val_df.label == 1).sum())},
        },
        "sanity_checks": sanity,
        "split_n_splits": 5 if args.split == "grupo" else None,
        "split_fold": 0 if args.split == "grupo" else None,
        "metrics_threshold": 0.5,
        "passo6": str(args.passo6.resolve()) if conservative else None,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "num_epochs_max": args.epocas,
        "early_stopping_patience": early_patience,
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
            "otimizador": f"AdamW(weight_decay={weight_decay})",
            "weight_decay": weight_decay,
            "no_decay_bias_and_1d": conservative,
            "lr_backbone": LR_BACKBONE, "lr_head": lr_head,
            "unfreeze_schedule": {str(k): v for k, v in unfreeze_schedule.items()},
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
    config["pos_weight"] = pos_weight.item()
    config["class_weights"] = weights.tolist()
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # --- 3. datasets e dataloaders ---
    train_ds = PCOSGenTrainDataset(args.img_dir, train_df, transform=train_transform)
    val_ds = PCOSGenTrainDataset(args.img_dir, val_df, transform=eval_transform)
    if conservative:
        save_augmentation_examples(args.img_dir, train_df, train_transform, eval_transform, exp_dir)
    train_loader = make_loader(train_ds, True, args.num_workers, device)
    val_loader = make_loader(val_ds, False, args.num_workers, device)
    print(f"\nBatches treino: {len(train_loader)} | Batches val: {len(val_loader)}")

    # --- 4. modelo, loss, otimizador, scheduler ---
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if adamw:
        model = build_resnet50(pretrained=True, freeze_backbone=False).to(device)
        # todos os parametros desde o inicio; os congelados ficam sem grad e o AdamW os ignora
        optimizer = torch.optim.AdamW(
            get_param_groups(model, LR_BACKBONE, lr_head,
                             weight_decay=weight_decay if conservative else None),
            weight_decay=weight_decay,
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
    print(f"Early stopping com paciencia de {early_patience} epocas ({criterio})")
    if adamw:
        print(f"AdamW: lr backbone={LR_BACKBONE}, lr cabeca={lr_head}, weight decay={weight_decay}")
        print(f"Descongelamento gradual (epoca -> estagios): {unfreeze_schedule}")
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
            if epoch in unfreeze_schedule:
                set_trainable_stages(model, unfreeze_schedule[epoch])
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

        if args.freeze_bn:
            freeze_batchnorm_stats(model)
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.label_smoothing, args.freeze_bn
        )
        val_metrics, val_labels, val_logits = evaluate(
            model, val_loader, criterion, device
        )
        train_loss, train_acc, train_auc, train_f1 = (
            train_metrics[k] for k in ("loss", "acc", "auc", "f1")
        )
        val_loss, val_acc, val_auc, val_f1 = (
            val_metrics[k] for k in ("loss", "acc", "auc", "f1")
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
            "lr_backbone": current_lr,
            "lr_head": head_lr,
            "trainable_params": trainable,
            "time_sec": elapsed,
        }
        for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
            for key in ("precision", "recall", "balanced_acc", "specificity", "tn", "fp", "fn", "tp"):
                row[f"{prefix}_{key}"] = metrics[key]
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
            diagnostics = probability_diagnostics(val_labels, val_logits)
            (exp_dir / "diagnostics").mkdir(exist_ok=True)
            diagnostic_row = {"epoch": epoch, **diagnostics}
            diagnostic_path = exp_dir / "diagnostics/probability_diagnostics.csv"
            first_diagnostic = not diagnostic_path.exists()
            with open(diagnostic_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=diagnostic_row.keys())
                if first_diagnostic:
                    writer.writeheader()
                writer.writerow(diagnostic_row)
            pd.DataFrame({
                "epoch": epoch, "imagePath": val_df["imagePath"], "label": val_labels.astype(int),
                "logit": val_logits, "prob_pcos": torch.sigmoid(torch.as_tensor(val_logits)).numpy(),
                "pred_label": (torch.sigmoid(torch.as_tensor(val_logits)).numpy() > 0.5).astype(int),
            }).to_csv(exp_dir / "diagnostics/best_validation_scores.csv", index=False)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_auc": val_auc,
                "val_f1": val_f1,
                "val_loss": val_loss,
                "lr": current_lr,
                "lr_backbone": current_lr,
                "lr_head": head_lr,
                "freeze_bn": args.freeze_bn,
                "val_metrics": val_metrics,
                "probability_diagnostics": diagnostics,
                "seed": SEED,
                "threshold": threshold,
                "threshold_origem": "Youden na validacao",
                "preproc": args.preproc,
            }, MODEL_PATH)
            print(f"  -> melhor modelo salvo ({criterio}: AUC={val_auc:.4f}, loss={val_loss:.4f}, "
                  f"limiar Youden={threshold:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_patience:
                print(f"\nEarly stopping: {early_patience} epocas sem melhora")
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
    if conservative:
        if args.debug_amostras:
            print("[DEBUG] Comparacao cientifica nao gerada para subconjuntos de debug.")
        else:
            compare_with_step6(args.passo6, exp_dir)


if __name__ == "__main__":
    main()
