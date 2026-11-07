"""
evaluate.py
Avaliacao do modelo treinado em multiplos conjuntos de teste.

Experimentos:
- Teste interno: PCOSGen-test (mesma fonte do treino)
- Validacao externa 1: Kaggle (Choudhari)
- Validacao externa 2: Figshare (Indirani)

Calcula metricas:
- AUC-ROC, F1, acuracia, precisao, sensibilidade, especificidade
- Matriz de confusao

Uso (--checkpoint e --out-dir sao obrigatorios, para nunca sobrescrever os
resultados do experimento de referencia em outputs/logs/ e outputs/scores/):

    python src/evaluate.py --checkpoint <best_model.pt> --out-dir <pasta> [--limiar ...]

Limiar de decisao (--limiar):
- "checkpoint" (padrao): o limiar salvo no checkpoint por train.py (Youden na
  validacao); checkpoints antigos, sem limiar salvo, usam 0,5
- um numero (ex.: 0.5)
- um arquivo .json com a chave "threshold" (gerado por calibrar_limiar.py)
O limiar nunca e' escolhido nos conjuntos de teste.

Pre-processamento: o mesmo do treino, lido do checkpoint ("preproc");
checkpoints antigos usam "rgb_resize".

Kaggle e Figshare: so' entram arquivos com extensao de imagem que o PIL
consegue abrir; os demais sao descartados e listados (relatorio, secao 3.3).

Duplicatas: se outputs/dedup/cross_overlap.csv existir (dedup.py), cada
imagem recebe a coluna duplicata_treino e as metricas sao calculadas tambem
sem as duplicatas do PCOSGen-train.

Salva em <pasta>:
- evaluation_results.csv: metricas por conjunto (todas as imagens e sem duplicatas)
- scores/scores_<conjunto>.csv: uma linha por imagem com image, label
  (0 = normal, 1 = PCOS), logit, prob_pcos, pred_label (prob_pcos > limiar),
  dist_treino e duplicata_treino
- arquivos_ilegiveis.csv: arquivos descartados
- avaliacao_info.json: checkpoint usado (caminho, SHA-256, epoca), limiar e origem
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix

from dataset import PCOSGenTrainDataset, load_pcosgen_labels
from transforms_config import get_eval_transform
from model import build_resnet50

# --- configuracao ---
# pastas do experimento de referencia: evaluate.py nao grava nelas
REFERENCE_DIRS = [Path("G:/tcc/outputs/logs"), Path("G:/tcc/outputs/scores"), Path("G:/tcc/outputs")]

# datasets
PCOSGEN_TEST_IMG_DIR = Path("G:/tcc/data/pcosgen/test/PCOSGen-test/images")
PCOSGEN_TEST_LABELS = Path("G:/tcc/data/pcosgen/test/class label.csv")

KAGGLE_DIR = Path("G:/tcc/data/kaggle/data")  # tem train/ e test/

FIGSHARE_DIR = Path("G:/tcc/data/figshare/PCOS")

DEDUP_PATH = Path("G:/tcc/outputs/dedup/cross_overlap.csv")

BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# DATASETS
# ============================================================

class FolderDataset(Dataset):
    """Dataset com pastas por classe (Kaggle e Figshare)."""

    def __init__(self, root_dirs, class_map, transform=None):
        """
        Args:
            root_dirs: dict {nome_classe: lista_de_pastas}.
            class_map: dict {nome_classe: rotulo_inteiro (0 ou 1)}.
            transform: transformacao a aplicar.
        """
        self.transform = transform
        self.samples = []
        self.skipped = []
        for class_name, folders in root_dirs.items():
            label = class_map[class_name]
            for folder in folders:
                for img_path in sorted(Path(folder).iterdir()):
                    if not img_path.is_file():
                        continue
                    if img_path.suffix.lower() not in IMG_EXTS:
                        self.skipped.append((img_path, label, "extensao"))
                        continue
                    try:
                        with Image.open(img_path) as im:
                            im.verify()
                    except Exception as e:
                        self.skipped.append((img_path, label, type(e).__name__))
                        continue
                    self.samples.append((img_path, label))
        if self.skipped:
            print(f"  [aviso] {len(self.skipped)} arquivos descartados (nao sao imagens validas)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @staticmethod
    def _id(path):
        # ultimos 3 niveis do caminho, ex.: train/infected/img_001.jpg
        return "/".join(path.parts[-3:])

    def sample_id(self, idx):
        return self._id(self.samples[idx][0])

    def skipped_rows(self):
        return [{"image": self._id(p), "label": label, "bytes": p.stat().st_size, "erro": erro}
                for p, label, erro in self.skipped]


# ============================================================
# AVALIACAO
# ============================================================

@torch.no_grad()
def predict(model, loader, device, threshold):
    """Logits, rotulos e predicoes (prob_pcos > threshold), na ordem do dataset."""
    model.eval()
    all_logits, all_preds, all_labels = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).long()
        all_logits.extend(logits.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_logits), np.array(all_preds)


def compute_metrics(labels, logits, preds, dataset_name, subset, threshold):
    auc = roc_auc_score(labels, logits)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # PCOS detectado
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # normal detectado
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # valor preditivo positivo

    print(f"\n{'=' * 60}")
    print(f"  {dataset_name} [{subset}]")
    print(f"{'=' * 60}")
    print(f"Amostras: {len(labels)} "
          f"(normal={int((labels == 0).sum())}, PCOS={int((labels == 1).sum())})")
    print(f"AUC-ROC      : {auc:.4f}")
    print(f"F1-score     : {f1:.4f}")
    print(f"Acuracia     : {acc:.4f}")
    print(f"Precisao     : {precision:.4f}")
    print(f"Sensibilidade: {sensitivity:.4f} (recall PCOS)")
    print(f"Especificidade: {specificity:.4f} (recall normal)")
    print(f"Matriz de confusao (limiar {threshold:.4f}):")
    print(f"               Predito 0   Predito 1")
    print(f"  Real 0 (N)   {tn:6d}      {fp:6d}")
    print(f"  Real 1 (P)   {fn:6d}      {tp:6d}")

    return {
        "dataset": dataset_name if subset == "todas" else f"{dataset_name} sem duplicatas",
        "subconjunto": subset,
        "n_samples": len(labels),
        "n_normal": int((labels == 0).sum()),
        "n_pcos": int((labels == 1).sum()),
        "auc_roc": auc,
        "f1_score": f1,
        "accuracy": acc,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": threshold,
    }


def evaluate_dataset(model, ds, device, dataset_name, conjunto, scores_path, threshold,
                     dedup, num_workers, n_skipped):
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=num_workers,
                        pin_memory=(device.type == "cuda"))
    labels, logits, preds = predict(model, loader, device, threshold)
    scores = pd.DataFrame({
        "image": [ds.sample_id(i) for i in range(len(ds))],
        "label": labels.astype(int),
        "logit": logits,
        "prob_pcos": 1 / (1 + np.exp(-logits)),
        "pred_label": preds.astype(int),
    })
    rows = [compute_metrics(labels, logits, preds, dataset_name, "todas", threshold)]
    if dedup is not None:
        d = dedup[dedup["conjunto"] == conjunto].drop_duplicates("image")
        scores = scores.merge(d[["image", "min_dist", "duplicata"]], on="image", how="left",
                              validate="many_to_one")
        scores = scores.rename(columns={"min_dist": "dist_treino", "duplicata": "duplicata_treino"})
        if scores["duplicata_treino"].isna().any():
            raise SystemExit(f"{conjunto}: imagens sem resultado em {DEDUP_PATH}; rode dedup.py de novo.")
        keep = ~scores["duplicata_treino"].astype(bool).values
        print(f"\n{conjunto}: {int((~keep).sum())} de {len(keep)} imagens sao duplicatas do PCOSGen-train")
        rows.append(compute_metrics(labels[keep], logits[keep], preds[keep], dataset_name,
                                    "sem duplicatas do treino", threshold))
    for r in rows:
        r["n_descartados"] = n_skipped
    scores.to_csv(scores_path, index=False)
    return rows


# ============================================================
# CARREGADORES DE CADA DATASET
# ============================================================

def load_pcosgen_test(transform):
    """PCOSGen-test: CSV com inversao de rotulo (Healthy=1 -> label=0)."""
    df = load_pcosgen_labels(PCOSGEN_TEST_LABELS)
    # filtra apenas imagens que existem
    img_files = {p.name for p in PCOSGEN_TEST_IMG_DIR.iterdir() if p.is_file()}
    df = df[df["imagePath"].isin(img_files)].reset_index(drop=True)
    return PCOSGenTrainDataset(PCOSGEN_TEST_IMG_DIR, df, transform=transform)


def load_kaggle(transform):
    """Kaggle: pastas train/test com infected/notinfected. Combinamos os dois."""
    root_dirs = {
        "notinfected": [
            KAGGLE_DIR / "train" / "notinfected",
            KAGGLE_DIR / "test" / "notinfected",
        ],
        "infected": [
            KAGGLE_DIR / "train" / "infected",
            KAGGLE_DIR / "test" / "infected",
        ],
    }
    class_map = {"notinfected": 0, "infected": 1}
    return FolderDataset(root_dirs, class_map, transform=transform)


def load_figshare(transform):
    """Figshare: pastas infected/noninfected."""
    root_dirs = {
        "noninfected": [FIGSHARE_DIR / "noninfected"],
        "infected": [FIGSHARE_DIR / "infected"],
    }
    class_map = {"noninfected": 0, "infected": 1}
    return FolderDataset(root_dirs, class_map, transform=transform)


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Avalia um checkpoint no PCOSGen-test, Kaggle e Figshare.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="arquivo best_model.pt")
    parser.add_argument("--out-dir", type=Path, required=True, help="pasta nova para os resultados")
    parser.add_argument("--limiar", default="checkpoint",
                        help='"checkpoint" (padrao), um numero ou um .json com a chave "threshold"')
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--debug-amostras", type=int, default=0,
                        help="so' para teste rapido: avalia as N primeiras imagens de cada conjunto")
    return parser.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_threshold(arg, checkpoint):
    if arg == "checkpoint":
        if "threshold" in checkpoint:
            return float(checkpoint["threshold"]), checkpoint.get("threshold_origem", "checkpoint")
        return 0.5, "padrao 0,5 (checkpoint sem limiar salvo)"
    path = Path(arg)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data["threshold"]), f"{data.get('origem', 'arquivo')} ({path})"
    return float(arg), "informado na linha de comando"


def main():
    args = parse_args()
    MODEL_PATH = args.checkpoint.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir in {d.resolve() for d in REFERENCE_DIRS}:
        raise SystemExit(f"{out_dir} guarda resultados do experimento de referencia; use outra pasta.")
    RESULTS_PATH = out_dir / "evaluation_results.csv"
    SCORES_DIR = out_dir / "scores"
    if RESULTS_PATH.exists() or SCORES_DIR.exists():
        raise SystemExit(f"{out_dir} ja tem resultados de avaliacao; escolha outra pasta.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- carrega modelo treinado ---
    print(f"\nCarregando modelo de: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    print(f"Modelo treinado na epoca {checkpoint['epoch']}, val_auc={checkpoint['val_auc']:.4f}")
    threshold, threshold_origem = resolve_threshold(args.limiar, checkpoint)
    preproc = checkpoint.get("preproc", "rgb_resize")
    print(f"Limiar de decisao: {threshold:.6f} ({threshold_origem})")
    print(f"Pre-processamento: {preproc}")

    dedup = pd.read_csv(DEDUP_PATH) if DEDUP_PATH.exists() else None
    if dedup is None:
        print(f"[aviso] {DEDUP_PATH} nao existe: metricas sem duplicatas nao serao calculadas")

    out_dir.mkdir(parents=True, exist_ok=True)
    transform = get_eval_transform(preproc)
    info = {
        "data_avaliacao": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(MODEL_PATH),
        "checkpoint_sha256": sha256(MODEL_PATH),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_val_auc": checkpoint["val_auc"],
        "threshold": threshold,
        "threshold_origem": threshold_origem,
        "preproc": preproc,
        "classes": {"0": "normal", "1": "PCOS (classe positiva)"},
        "eval_transform": repr(transform),
        "dedup": str(DEDUP_PATH) if dedup is not None else None,
        "debug_amostras": args.debug_amostras,
        "device": str(device),
    }

    model = build_resnet50(pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def limitar(ds):
        if args.debug_amostras:
            n = min(args.debug_amostras, len(ds))
            if isinstance(ds, FolderDataset):
                ds.samples = ds.samples[:n // 2] + ds.samples[-(n - n // 2):]
            else:
                ds.labels_df = ds.labels_df.head(n)
        return ds

    # --- avalia em cada dataset ---
    results = []
    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n>>> Carregando PCOSGen-test (validacao interna)...")
    ds = limitar(load_pcosgen_test(transform))
    results += evaluate_dataset(model, ds, device, "PCOSGen-test (interno)", "PCOSGen-test",
                                SCORES_DIR / "scores_pcosgen_test.csv", threshold, dedup,
                                args.num_workers, 0)

    print("\n>>> Carregando Kaggle (validacao externa 1)...")
    ds = load_kaggle(transform)
    skipped = [dict(conjunto="Kaggle", **r) for r in ds.skipped_rows()]
    ds = limitar(ds)
    results += evaluate_dataset(model, ds, device, "Kaggle (externo 1)", "Kaggle",
                                SCORES_DIR / "scores_kaggle.csv", threshold, dedup,
                                args.num_workers, len(ds.skipped))

    print("\n>>> Carregando Figshare (validacao externa 2)...")
    ds = load_figshare(transform)
    skipped += [dict(conjunto="Figshare", **r) for r in ds.skipped_rows()]
    ds = limitar(ds)
    results += evaluate_dataset(model, ds, device, "Figshare (externo 2)", "Figshare",
                                SCORES_DIR / "scores_figshare.csv", threshold, dedup,
                                args.num_workers, len(ds.skipped))

    # --- salva CSV ---
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    pd.DataFrame(skipped, columns=["conjunto", "image", "label", "bytes", "erro"]).to_csv(
        out_dir / "arquivos_ilegiveis.csv", index=False)
    info["n_imagens"] = {r["dataset"]: r["n_samples"] for r in results}
    info["n_descartados"] = len(skipped)
    with open(out_dir / "avaliacao_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"\nArquivos descartados (nao sao imagens validas): {len(skipped)}")

    # --- resumo final ---
    print(f"\n{'=' * 70}")
    print("RESUMO")
    print(f"{'=' * 70}")
    print(f"{'Dataset':<45} {'n':>6} {'AUC':>7} {'F1':>7} {'Acc':>7}")
    print(f"{'-' * 70}")
    for r in results:
        print(f"{r['dataset']:<45} {r['n_samples']:>6} {r['auc_roc']:>7.4f} "
              f"{r['f1_score']:>7.4f} {r['accuracy']:>7.4f}")

    print(f"\nResultados salvos em: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
