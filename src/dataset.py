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
    if "Healthy" not in df.columns:
        raise ValueError("Rotulos devem conter a coluna Healthy")
    if not df["Healthy"].isin([0, 1]).all():
        raise ValueError("Healthy deve conter apenas 0 ou 1, sem valores ausentes ou fracionarios")
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


def validate_train_val_split(train_df, val_df, groups_df=None):
    """Valida rotulos, IDs e ausencia de vazamento; retorna contagens por lado.

    groups_df, quando fornecido, deve mapear imagePath unico para group_id
    nao nulo e cobrir todas as imagens de treino e validacao.
    """
    counts = {}
    images = {}
    for side, df in [("train", train_df), ("val", val_df)]:
        if not {"imagePath", "label"}.issubset(df.columns):
            raise ValueError(f"{side}: colunas imagePath e label sao obrigatorias")
        if not df["label"].isin([0, 1]).all() or set(df["label"]) != {0, 1}:
            raise ValueError(f"{side}: label deve conter ambas as classes {{0, 1}}, sem valores invalidos")
        if df["imagePath"].isna().any():
            raise ValueError(f"{side}: imagePath nao pode conter valores nulos")
        if df["imagePath"].duplicated().any():
            raise ValueError(f"{side}: imagePath deve ser unico (imagens duplicadas)")
        images[side] = set(df["imagePath"])
        counts[f"{side}_images"] = len(df)
        counts[f"{side}_label_counts"] = {label: int((df["label"] == label).sum()) for label in (0, 1)}

    overlap = images["train"] & images["val"]
    if overlap:
        raise ValueError(f"Vazamento: {len(overlap)} imagePath compartilhados entre train e val")
    counts["image_overlap"] = 0

    if groups_df is not None:
        if not {"imagePath", "group_id"}.issubset(groups_df.columns):
            raise ValueError("groups_df: colunas imagePath e group_id sao obrigatorias")
        if groups_df[["imagePath", "group_id"]].isna().any().any():
            raise ValueError("groups_df: imagePath e group_id nao podem conter valores nulos")
        if groups_df["imagePath"].duplicated().any():
            raise ValueError("groups_df: imagePath deve ser unico (IDs duplicados)")
        missing = (images["train"] | images["val"]) - set(groups_df["imagePath"])
        if missing:
            raise ValueError(f"groups_df: cobertura incompleta, faltam {len(missing)} imagePath")
        group_ids = {
            side: set(groups_df.loc[groups_df["imagePath"].isin(ids), "group_id"])
            for side, ids in images.items()
        }
        overlap = group_ids["train"] & group_ids["val"]
        if overlap:
            raise ValueError(f"Vazamento: {len(overlap)} group_id compartilhados entre train e val")
        counts["train_groups"] = len(group_ids["train"])
        counts["val_groups"] = len(group_ids["val"])
        counts["group_overlap"] = 0
    return counts


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