"""
calibrar_limiar.py
Calcula o limiar de Youden (sensibilidade + especificidade - 1 maximo) na
VALIDACAO de um checkpoint ja treinado, sem retreinar (relatorio tecnico,
secao 3.4 e passo 1 da ablacao). O limiar nunca e' escolhido no teste.

A validacao e' refeita com a mesma divisao do treino daquele checkpoint
(lida do config.json do experimento; padrao: aleatoria 80/20, seed 42).

Uso:
    python src/calibrar_limiar.py --checkpoint <best_model.pt> --out <limiar.json>

Depois:
    python src/evaluate.py --checkpoint <best_model.pt> --limiar <limiar.json> --out-dir <pasta>
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import PCOSGenTrainDataset, load_pcosgen_train_labels, split_train_val, split_train_val_grupos
from model import build_resnet50
from train import IMG_DIR, LABELS_PATH, SEED, youden_threshold
from transforms_config import get_eval_transform


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="arquivo .json de saida")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def sens_spec(labels, probs, threshold):
    pred = probs > threshold
    sens = (pred & (labels == 1)).sum() / (labels == 1).sum()
    spec = (~pred & (labels == 0)).sum() / (labels == 0).sum()
    return float(sens), float(spec)


@torch.no_grad()
def main():
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} ja existe.")
    ckpt_path = args.checkpoint.resolve()
    exp_dir = ckpt_path.parent.parent
    cfg_path = exp_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    split = cfg.get("split", "aleatorio")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    preproc = checkpoint.get("preproc", cfg.get("preproc", "rgb_resize"))

    df = load_pcosgen_train_labels(LABELS_PATH)
    if split == "grupo":
        grupos = pd.read_csv(cfg["grupos"])
        _, val_df = split_train_val_grupos(df, grupos["group_id"].values, seed=SEED)
    else:
        _, val_df = split_train_val(df, val_size=0.2, seed=SEED)
    saved_val = exp_dir / "splits" / "pcosgen_train_validacao.csv"
    if saved_val.exists():
        ref = pd.read_csv(saved_val)
        if not (ref["imagePath"].astype(str).values == val_df["imagePath"].astype(str).values).all():
            raise SystemExit(f"Validacao refeita difere de {saved_val}.")
        print(f"Validacao identica a usada no treino ({saved_val})")

    model = build_resnet50(pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(PCOSGenTrainDataset(IMG_DIR, val_df, transform=get_eval_transform(preproc)),
                        batch_size=32, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))
    labels, logits = [], []
    for images, y in loader:
        logits.extend(model(images.to(device)).squeeze(1).cpu().numpy())
        labels.extend(y.numpy())
    labels, logits = np.array(labels), np.array(logits, dtype=np.float32)

    threshold = youden_threshold(labels, logits)
    probs = torch.sigmoid(torch.as_tensor(logits)).numpy()
    s05, e05 = sens_spec(labels, probs, 0.5)
    st, et = sens_spec(labels, probs, threshold)
    out = {
        "threshold": threshold,
        "origem": "Youden na validacao (calibrar_limiar.py)",
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "split_validacao": split,
        "preproc": preproc,
        "n_validacao": int(len(labels)),
        "validacao_limiar_0.5": {"sensibilidade": s05, "especificidade": e05},
        "validacao_limiar_youden": {"sensibilidade": st, "especificidade": et},
        "data": datetime.now().isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
