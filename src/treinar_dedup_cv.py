"""
treinar_dedup_cv.py
Treino e avaliacao na base PCOSGen deduplicada do PelviNeXt (Bhuiyan et al.,
2026; 225 imagens, pHash <= 14), com validacao cruzada estratificada em 5 folds.

- Configuracao do passo 3 da ablacao (escala de cinza + PadToSquare, augmentation
  e otimizacao do PR #1, early stopping por val_auc), sem mudar hiperparametros.
- A base mistura imagens do PCOSGen-train e do PCOSGen-test, entao nao ha teste
  interno separado: o desempenho interno e' a validacao de cada fold.
- Cada modelo de fold e' avaliado no Kaggle e no Figshare (todas as imagens e
  sem as quase-copias das 225 imagens, pHash <= 6 com flip/rotacoes, como no
  dedup.py). Limiares: 0,5 e Youden da validacao do fold.
- Nenhum arquivo de imagem ou rotulo e' alterado.

Uso:
    python src/treinar_dedup_cv.py [--out-dir <pasta>] [--epocas N] [--sem-early-stopping]
                                   [--debug-amostras N]

Saidas em <out-dir> (padrao outputs/experimentos/dedup_pelvinext_cv_<data_hora>):
- fold<k>/: config.json, logs/train_log.csv, models/best_model.pt, splits/,
  validacao_scores.csv, scores_<conjunto>.csv
- metricas_por_fold.csv, metricas_resumo.csv (media e desvio entre folds)
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import train as T
from dataset import PCOSGenTrainDataset
from evaluate import load_kaggle, load_figshare, predict, compute_metrics
from model import build_resnet50
from transforms_config import get_train_transform, get_eval_transform

RAIZ = Path(__file__).resolve().parent.parent
DEDUP_DIR = RAIZ / "data" / "pcosgen_dedup" / "pcosgen-deduplicated"
HASH_DIR = RAIZ / "outputs" / "dedup"
N_FOLDS = 5
PREPROC, AUGMENT = "gray_pad", "pr1"
DUP_THRESHOLD = 6


def load_dedup():
    rows = [{"imagePath": f"{cls}/{p.name}", "label": lab}
            for cls, lab in [("Normal", 0), ("Abnormal", 1)]
            for p in sorted((DEDUP_DIR / cls).glob("*.jpg"))]
    return pd.DataFrame(rows)


def duplicate_flags(df):
    """Para Kaggle e Figshare: a imagem e' quase-copia (pHash <= 6) de alguma das 225?"""
    tr = np.load(HASH_DIR / "hashes_pcosgen_train.npz", allow_pickle=True)
    te = np.load(HASH_DIR / "hashes_pcosgen_test.npz", allow_pickle=True)
    idx = {n: tr["hashes"][i, 0] for i, n in enumerate(tr["imagePath"])}
    idx.update({n: te["hashes"][i, 0] for i, n in enumerate(te["image"])})
    names = [p.split("/", 1)[1] for p in df["imagePath"]]
    missing = [n for n in names if n not in idx]
    base = np.array([idx[n] for n in names if n in idx], dtype=np.uint64)
    flags = {}
    for conj, arq in [("Kaggle", "hashes_kaggle.npz"), ("Figshare", "hashes_figshare.npz")]:
        z = np.load(HASH_DIR / arq, allow_pickle=True)
        h, ok, ids = z["hashes"], z["ok"], z["image"]
        d = None
        for v in range(h.shape[1]):
            x = np.bitwise_count(np.bitwise_xor(h[:, v][:, None], base[None, :])).astype(np.int16)
            d = x if d is None else np.minimum(d, x)
        dmin = d.min(axis=1)
        flags[conj] = dict(zip(ids, (dmin <= DUP_THRESHOLD) & ok))
    return flags, missing


def train_fold(k, train_df, val_df, fold_dir, args, device):
    T.set_seed(T.SEED)
    (fold_dir / "models").mkdir(parents=True, exist_ok=True)
    (fold_dir / "logs").mkdir(exist_ok=True)
    (fold_dir / "splits").mkdir(exist_ok=True)
    train_df.to_csv(fold_dir / "splits" / "treino.csv", index=False)
    val_df.to_csv(fold_dir / "splits" / "validacao.csv", index=False)
    model_path, log_path = fold_dir / "models" / "best_model.pt", fold_dir / "logs" / "train_log.csv"

    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_df["label"].values)
    pos_weight = torch.tensor(w[1] / w[0]).to(device)
    train_ds = PCOSGenTrainDataset(DEDUP_DIR, train_df, transform=get_train_transform(PREPROC, AUGMENT))
    val_ds = PCOSGenTrainDataset(DEDUP_DIR, val_df, transform=get_eval_transform(PREPROC))
    train_loader = T.make_loader(train_ds, True, args.num_workers, device)
    val_loader = T.make_loader(val_ds, False, args.num_workers, device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = build_resnet50(pretrained=True, freeze_backbone=True).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=T.HEAD_LR, weight_decay=T.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                           patience=5, min_lr=1e-6)
    config = {"fold": k, "inicio": datetime.now().isoformat(timespec="seconds"),
              "base": str(DEDUP_DIR), "n_treino": len(train_df), "n_validacao": len(val_df),
              "dist_treino": train_df["label"].value_counts().to_dict(),
              "dist_validacao": val_df["label"].value_counts().to_dict(),
              "preproc": PREPROC, "augment": AUGMENT, "otimizacao": "pr1 (passo 3)",
              "pos_weight": pos_weight.item(), "epocas_max": args.epocas,
              "early_stopping": ("desativado" if args.sem_early_stopping
                                 else f"{T.EARLY_STOPPING_PATIENCE} epocas sem melhora da val_auc")}

    # epoca 0: modelo antes de qualquer atualizacao (so' registro; nao entra na selecao).
    # O "treino" da epoca 0 e' avaliado sem augmentation e em modo eval; nas demais
    # epocas o valor de treino e' a media durante a epoca, com augmentation.
    train_eval_loader = T.make_loader(
        PCOSGenTrainDataset(DEDUP_DIR, train_df, transform=get_eval_transform(PREPROC)),
        False, args.num_workers, device)
    tr0 = T.evaluate(model, train_eval_loader, criterion, device)
    va0 = T.evaluate(model, val_loader, criterion, device)
    row0 = {"epoch": 0, "train_loss": tr0[0], "train_acc": tr0[1], "train_auc": tr0[2],
            "train_f1": tr0[3], "val_loss": va0[0], "val_acc": va0[1], "val_auc": va0[2],
            "val_f1": va0[3], "lr": optimizer.param_groups[0]["lr"], "time_sec": 0.0}
    with open(log_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=row0.keys())
        wr.writeheader()
        wr.writerow(row0)
    print(f"[fold {k}] Ep   0 | trn loss={tr0[0]:.4f} auc={tr0[2]:.3f} | "
          f"val loss={va0[0]:.4f} auc={va0[2]:.3f} | antes do treino", flush=True)

    best, patience = None, 0
    for epoch in range(1, args.epocas + 1):
        if epoch == T.HEAD_ONLY_EPOCHS + 1:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=T.FINE_TUNE_LR, weight_decay=T.WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                                   patience=5, min_lr=1e-6)
        t0 = time.time()
        tr_loss, tr_acc, tr_auc, tr_f1 = T.train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_auc, va_f1, va_labels, va_logits = T.evaluate(model, val_loader, criterion, device)
        scheduler.step(va_auc)
        row = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "train_auc": tr_auc,
               "train_f1": tr_f1, "val_loss": va_loss, "val_acc": va_acc, "val_auc": va_auc,
               "val_f1": va_f1, "lr": optimizer.param_groups[0]["lr"], "time_sec": time.time() - t0}
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=row.keys()).writerow(row)
        print(f"[fold {k}] Ep {epoch:3d} | trn loss={tr_loss:.4f} auc={tr_auc:.3f} | "
              f"val loss={va_loss:.4f} auc={va_auc:.3f} | {row['time_sec']:.1f}s", flush=True)
        if best is None or va_auc > best:
            best, patience = va_auc, 0
            thr = T.youden_threshold(va_labels, va_logits)
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_auc": va_auc,
                        "val_loss": va_loss, "threshold": thr, "preproc": PREPROC}, model_path)
            pd.DataFrame({"imagePath": val_df["imagePath"], "label": va_labels.astype(int),
                          "logit": va_logits}).to_csv(fold_dir / "validacao_scores.csv", index=False)
        else:
            patience += 1
            if not args.sem_early_stopping and patience >= T.EARLY_STOPPING_PATIENCE:
                print(f"[fold {k}] early stopping na epoca {epoch}", flush=True)
                break
    ck = torch.load(model_path, map_location="cpu", weights_only=True)
    config.update({"fim": datetime.now().isoformat(timespec="seconds"), "epocas_executadas": epoch,
                   "melhor_epoca": ck["epoch"], "melhor_val_auc": ck["val_auc"],
                   "limiar_youden": ck["threshold"]})
    (fold_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return ck


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--epocas", type=int, default=T.NUM_EPOCHS)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--sem-early-stopping", action="store_true",
                   help="treina todas as --epocas; o modelo usado continua sendo o da maior val_auc")
    p.add_argument("--debug-amostras", type=int, default=0)
    args = p.parse_args()
    out = (args.out_dir or RAIZ / "outputs" / "experimentos"
           / f"dedup_pelvinext_cv_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    if (out / "metricas_por_fold.csv").exists():
        raise SystemExit(f"{out} ja tem resultados.")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Pasta: {out}\nDevice: {device}")

    df = load_dedup()
    print(f"Base deduplicada: {len(df)} imagens {df['label'].value_counts().to_dict()}")
    flags, missing = duplicate_flags(df)
    print(f"Sem hash para {len(missing)} imagem(ns) da base deduplicada: {missing}")

    transform = get_eval_transform(PREPROC)
    externos = {"Kaggle": load_kaggle(transform), "Figshare": load_figshare(transform)}
    for nome, ds in externos.items():
        if args.debug_amostras:
            n = args.debug_amostras
            ds.samples = ds.samples[:n // 2] + ds.samples[-(n - n // 2):]
        dup = np.array([flags[nome].get(ds.sample_id(i), False) for i in range(len(ds))])
        print(f"{nome}: {len(ds)} imagens, {int(dup.sum())} quase-copias das 225")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=T.SEED)
    linhas = []
    for k, (tr_idx, va_idx) in enumerate(skf.split(df, df["label"])):
        fold_dir = out / f"fold{k}"
        train_df, val_df = df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)
        print(f"\n===== Fold {k}: treino {len(train_df)} | validacao {len(val_df)} =====", flush=True)
        ck = train_fold(k, train_df, val_df, fold_dir, args, device)
        model = build_resnet50(pretrained=False).to(device)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        v = pd.read_csv(fold_dir / "validacao_scores.csv")
        for lim_nome, lim in [("0,5", 0.5), ("Youden (validacao)", ck["threshold"])]:
            pred = (torch.sigmoid(torch.tensor(v["logit"].values, dtype=torch.float32)).numpy() > lim).astype(int)
            m = compute_metrics(v["label"].values, v["logit"].values, pred,
                                "Validacao do fold (interno)", "todas", lim)
            linhas.append({"fold": k, "limiar_tipo": lim_nome, "melhor_epoca": ck["epoch"], **m})
        for nome, ds in externos.items():
            loader = torch.utils.data.DataLoader(ds, batch_size=T.BATCH_SIZE, num_workers=args.num_workers,
                                                 pin_memory=(device.type == "cuda"))
            labels, logits, _ = predict(model, loader, device, 0.5)
            dup = np.array([flags[nome].get(ds.sample_id(i), False) for i in range(len(ds))])
            pd.DataFrame({"image": [ds.sample_id(i) for i in range(len(ds))], "label": labels,
                          "logit": logits, "duplicata_da_base": dup}).to_csv(
                fold_dir / f"scores_{nome.lower()}.csv", index=False)
            probs = torch.sigmoid(torch.tensor(logits, dtype=torch.float32)).numpy()
            for lim_nome, lim in [("0,5", 0.5), ("Youden (validacao)", ck["threshold"])]:
                pred = (probs > lim).astype(int)
                for sub, keep in [("todas", np.ones(len(dup), bool)), ("sem duplicatas da base", ~dup)]:
                    m = compute_metrics(labels[keep], logits[keep], pred[keep], nome, sub, lim)
                    linhas.append({"fold": k, "limiar_tipo": lim_nome, "melhor_epoca": ck["epoch"], **m})
        pd.DataFrame(linhas).to_csv(out / "metricas_por_fold.csv", index=False)

    r = pd.DataFrame(linhas)
    cols = ["auc_roc", "accuracy", "sensitivity", "specificity", "f1_score", "precision"]
    resumo = r.groupby(["dataset", "subconjunto", "limiar_tipo"], sort=False)[cols].agg(["mean", "std"])
    resumo.columns = [f"{a}_{b}" for a, b in resumo.columns]
    resumo.reset_index().to_csv(out / "metricas_resumo.csv", index=False)
    print("\n" + resumo.round(3).to_string())
    print(f"\nConcluido: {out}")


if __name__ == "__main__":
    main()
