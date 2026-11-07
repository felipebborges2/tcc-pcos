"""
dataset.py
Dataset PyTorch para o PCOSGen (train e test).

Convencao de rotulos no projeto: 0 = normal, 1 = PCOS.
PCOSGen usa Healthy=1 (normal) e Healthy=0 (PCOS), entao invertemos no load.
"""

from pathlib import Path
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


class PCOSGenTrainDataset(Dataset):
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
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def sample_id(self, idx):
        return str(self.labels_df.iloc[idx]["imagePath"])


def load_pcosgen_labels(labels_path):
    """Le rotulos do PCOSGen (xlsx no train, csv no test). Healthy=1 -> label 0 (normal)."""
    labels_path = Path(labels_path)
    if labels_path.suffix.lower() == ".csv":
        df = pd.read_csv(labels_path)
    else:
        df = pd.read_excel(labels_path)
    df.columns = [c.strip() for c in df.columns]
    df["label"] = 1 - df["Healthy"].astype(int)
    return df[["imagePath", "label"]]


def load_pcosgen_train_labels(labels_path):
    return load_pcosgen_labels(labels_path)


def split_train_val(df, val_size=0.2, seed=42):
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df["label"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def split_train_val_grupos(df, groups, n_splits=5, fold=0, seed=42):
    """
    Divide treino/validacao garantindo que duplicatas (mesmo group_id, gerado
    por dedup.py) fiquem sempre do mesmo lado. n_splits=5 -> ~20% de validacao.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(sgkf.split(df, df["label"], groups=groups))
    train_idx, val_idx = splits[fold]
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
    )


if __name__ == "__main__":
    LABELS_PATH = Path("G:/tcc/data/pcosgen/train/PCOSGen-train/PCOSGen-train/class_label.xlsx")
    IMG_DIR = Path("G:/tcc/data/pcosgen/train/PCOSGen-train/PCOSGen-train/images")

    print("Carregando rotulos...")
    df = load_pcosgen_train_labels(LABELS_PATH)
    print(f"Total: {len(df)} imagens")
    print("Distribuicao (convencao do projeto):")
    print(df["label"].value_counts().to_string())
    print("  label=0 -> normal")
    print("  label=1 -> PCOS")

    print("\nDividindo treino/validacao (80/20 estratificado)...")
    train_df, val_df = split_train_val(df)
    print(f"Treino: {len(train_df)} imagens")
    print(train_df["label"].value_counts().to_string())
    print(f"\nValidacao: {len(val_df)} imagens")
    print(val_df["label"].value_counts().to_string())

    print("\nTeste de carregamento de uma amostra...")
    ds = PCOSGenTrainDataset(IMG_DIR, train_df)
    img, label = ds[0]
    print(f"Imagem: {img.size}, tipo: {type(img).__name__}")
    print(f"Rotulo: {label}")