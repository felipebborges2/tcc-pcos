"""
evaluate.py
Avaliacao do modelo treinado em multiplos conjuntos de teste.

Experimentos:
- Teste interno: PCOSGen-test (mesma fonte do treino)
- Validacao externa 1: Kaggle (Choudhari)
- Validacao externa 2: Figshare (Indirani)

Calcula metricas:
- AUC-ROC, F1, acuracia, sensibilidade, especificidade
- Matriz de confusao

Uso (os dois argumentos sao obrigatorios, para nunca sobrescrever os
resultados do experimento de referencia em outputs/logs/ e outputs/scores/):

    python src/evaluate.py --checkpoint <best_model.pt> --out-dir <pasta>

Salva em <pasta>:
- evaluation_results.csv: metricas por conjunto (mesmas colunas de antes + precisao)
- scores/scores_<conjunto>.csv: uma linha por imagem com image, label
  (0 = normal, 1 = PCOS), logit, prob_pcos e pred_label (prob_pcos > 0.5)
- arquivos_ilegiveis.csv: arquivos que o PIL nao abre e que sao avaliados como
  imagem preta (mesmo tratamento usado nos resultados de referencia)
- avaliacao_info.json: checkpoint usado (caminho, SHA-256, epoca, val_auc)
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
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, classification_report,
)

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

BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 42
THRESHOLD = 0.5  # fixo; o mesmo dos resultados de referencia

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# DATASETS — um pra cada estrutura
# ============================================================

class CsvLabeledDataset(Dataset):
    """Dataset com rotulos em CSV/Excel (caso do PCOSGen)."""
    
    def __init__(self, img_dir, labels_df, transform=None):
        self.img_dir = Path(img_dir)
        self.labels_df = labels_df.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_path = self.img_dir / str(row["imagePath"])
        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label

    def sample_id(self, idx):
        return str(self.labels_df.iloc[idx]["imagePath"])


class FolderDataset(Dataset):
    """Dataset com pastas por classe (caso do Kaggle e Figshare)."""
    
    def __init__(self, root_dirs, class_map, transform=None):
        """
        Args:
            root_dirs: dict {nome_classe: lista_de_pastas}.
                       Exemplo: {"infected": [path1, path2], "notinfected": [path3]}
            class_map: dict {nome_classe: rotulo_inteiro (0 ou 1)}.
            transform: transformacao a aplicar.
        """
        self.transform = transform
        self.samples = []
        for class_name, folders in root_dirs.items():
            label = class_map[class_name]
            for folder in folders:
                folder = Path(folder)
                for img_path in folder.iterdir():
                    if img_path.is_file():
                        self.samples.append((img_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # imagens corrompidas (caso do Kaggle): retorna preto
            image = Image.new("RGB", (224, 224))
        if self.transform:
            image = self.transform(image)
        return image, label

    def sample_id(self, idx):
        # ultimos 3 niveis do caminho, ex.: train/infected/img_001.jpg
        return "/".join(self.samples[idx][0].parts[-3:])

    def unreadable_files(self):
        """Arquivos que caem no except de __getitem__ (avaliados como imagem preta)."""
        bad = []
        for i, (img_path, label) in enumerate(self.samples):
            try:
                Image.open(img_path).convert("RGB")
            except Exception as e:
                bad.append({"image": self.sample_id(i), "label": label,
                            "bytes": img_path.stat().st_size, "erro": type(e).__name__})
        return bad


# ============================================================
# AVALIACAO
# ============================================================

@torch.no_grad()
def evaluate_dataset(model, loader, device, dataset_name, scores_path=None):
    """Avalia modelo num dataset e retorna metricas.

    Se scores_path for informado, salva o score de cada imagem nesse CSV
    (requer loader sem shuffle, para manter a ordem do dataset).
    """
    model.eval()
    all_logits = []
    all_preds = []
    all_labels = []
    
    for images, labels in loader:
        images = images.to(device)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits)
        preds = (probs > THRESHOLD).long()
        
        all_logits.extend(logits.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
    
    all_logits = np.array(all_logits)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if scores_path is not None:
        ds = loader.dataset
        pd.DataFrame({
            "image": [ds.sample_id(i) for i in range(len(ds))],
            "label": all_labels.astype(int),
            "logit": all_logits,
            "prob_pcos": 1 / (1 + np.exp(-all_logits)),
            "pred_label": all_preds.astype(int),
        }).to_csv(scores_path, index=False)

    auc =roc_auc_score(all_labels, all_logits)
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # PCOS detectado
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # normal detectado
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # valor preditivo positivo
    
    print(f"\n{'=' * 60}")
    print(f"  {dataset_name}")
    print(f"{'=' * 60}")
    print(f"Amostras: {len(all_labels)} "
          f"(normal={int((all_labels == 0).sum())}, "
          f"PCOS={int((all_labels == 1).sum())})")
    print(f"AUC-ROC      : {auc:.4f}")
    print(f"F1-score     : {f1:.4f}")
    print(f"Acuracia     : {acc:.4f}")
    print(f"Precisao     : {precision:.4f}")
    print(f"Sensibilidade: {sensitivity:.4f} (recall PCOS)")
    print(f"Especificidade: {specificity:.4f} (recall normal)")
    print(f"Matriz de confusao:")
    print(f"               Predito 0   Predito 1")
    print(f"  Real 0 (N)   {tn:6d}      {fp:6d}")
    print(f"  Real 1 (P)   {fn:6d}      {tp:6d}")
    
    return {
        "dataset": dataset_name,
        "n_samples": len(all_labels),
        "n_normal": int((all_labels == 0).sum()),
        "n_pcos": int((all_labels == 1).sum()),
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
        "threshold": THRESHOLD,
    }


# ============================================================
# CARREGADORES DE CADA DATASET
# ============================================================

def load_pcosgen_test(transform):
    """PCOSGen-test: CSV com inversao de rotulo (Healthy=1 -> label=0)."""
    df = pd.read_csv(PCOSGEN_TEST_LABELS)
    df.columns = [c.strip() for c in df.columns]
    df["label"] = 1 - df["Healthy"].astype(int)
    df = df[["imagePath", "label"]]
    # filtra apenas imagens que existem
    img_files = {p.name for p in PCOSGEN_TEST_IMG_DIR.iterdir() if p.is_file()}
    df = df[df["imagePath"].isin(img_files)].reset_index(drop=True)
    return CsvLabeledDataset(PCOSGEN_TEST_IMG_DIR, df, transform=transform)


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
    return parser.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    print(f"Modelo treinado na epoca {checkpoint['epoch']}, val_auc={checkpoint['val_auc']:.4f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "data_avaliacao": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(MODEL_PATH),
        "checkpoint_sha256": sha256(MODEL_PATH),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_val_auc": checkpoint["val_auc"],
        "threshold": THRESHOLD,
        "classes": {"0": "normal", "1": "PCOS (classe positiva)"},
        "eval_transform": repr(get_eval_transform()),
        "device": str(device),
    }
    
    model = build_resnet50(pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # --- transform de avaliacao (sem augmentation) ---
    transform = get_eval_transform()
    
    # --- avalia em cada dataset ---
    results = []
    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n>>> Carregando PCOSGen-test (validacao interna)...")
    ds = load_pcosgen_test(transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    results.append(evaluate_dataset(model, loader, device, "PCOSGen-test (interno)",
                                   SCORES_DIR / "scores_pcosgen_test.csv"))
    
    print("\n>>> Carregando Kaggle (validacao externa 1)...")
    ds = load_kaggle(transform)
    unreadable = [dict(conjunto="Kaggle", **r) for r in ds.unreadable_files()]
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    results.append(evaluate_dataset(model, loader, device, "Kaggle (externo 1)",
                                   SCORES_DIR / "scores_kaggle.csv"))
    
    print("\n>>> Carregando Figshare (validacao externa 2)...")
    ds = load_figshare(transform)
    unreadable += [dict(conjunto="Figshare", **r) for r in ds.unreadable_files()]
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    results.append(evaluate_dataset(model, loader, device, "Figshare (externo 2)",
                                   SCORES_DIR / "scores_figshare.csv"))
    
    # --- salva CSV ---
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    pd.DataFrame(unreadable, columns=["conjunto", "image", "label", "bytes", "erro"]).to_csv(
        out_dir / "arquivos_ilegiveis.csv", index=False)
    info["n_imagens"] = {r["dataset"]: r["n_samples"] for r in results}
    info["n_ilegiveis_imagem_preta"] = len(unreadable)
    with open(out_dir / "avaliacao_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"\nArquivos ilegiveis (avaliados como imagem preta): {len(unreadable)}")
    
    # --- resumo final ---
    print(f"\n{'=' * 60}")
    print("RESUMO COMPARATIVO")
    print(f"{'=' * 60}")
    print(f"{'Dataset':<30} {'AUC':>8} {'F1':>8} {'Acc':>8}")
    print(f"{'-' * 60}")
    for r in results:
        print(f"{r['dataset']:<30} {r['auc_roc']:>8.4f} {r['f1_score']:>8.4f} {r['accuracy']:>8.4f}")
    
    interno_auc = results[0]["auc_roc"]
    print(f"\nGeneralization gap:")
    for r in results[1:]:
        gap = interno_auc - r["auc_roc"]
        print(f"  {r['dataset']}: queda de {gap:.4f} pontos de AUC ({gap*100:.1f}%)")
    
    print(f"\nResultados salvos em: {RESULTS_PATH}")


if __name__ == "__main__":
    main()