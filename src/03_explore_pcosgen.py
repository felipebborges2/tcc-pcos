"""
03_explore_pcosgen.py
Sanity check do dataset PCOSGen (train + test).
- Le rotulos do CSV/Excel
- Conta imagens por classe
- Verifica integridade
- Coleta resolucoes
- Salva amostras visuais

IMPORTANTE: Convencao do PCOSGen e' Healthy=1 (normal), Healthy=0 (PCOS).
"""

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import random

# --- configuracao ---
TRAIN_IMG_DIR = Path("G:/tcc/data/pcosgen/train/PCOSGen-train/PCOSGen-train/images")
TRAIN_LABELS = Path("G:/tcc/data/pcosgen/train/PCOSGen-train/PCOSGen-train/class_label.xlsx")
TEST_IMG_DIR = Path("G:/tcc/data/pcosgen/test/PCOSGen-test/images")
TEST_LABELS = Path("G:/tcc/data/pcosgen/test/class label.csv")
OUTPUT_DIR = Path("G:/tcc/outputs/figures")
SAMPLES_PER_CLASS = 8

print("=" * 60)
print("SANITY CHECK - PCOSGen Dataset (Handa et al.)")
print("=" * 60)

def load_labels(path):
    """Le rotulos de CSV ou XLSX."""
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    # normaliza nomes de coluna (alguns arquivos tem espacos)
    df.columns = [c.strip() for c in df.columns]
    return df

def analyze_split(name, img_dir, labels_df, label_col, img_col):
    """Analisa um split (train ou test)."""
    print(f"\n--- SPLIT: {name} ---")
    print(f"Total de linhas no arquivo de rotulos: {len(labels_df)}")
    
    # distribuicao de classes
    print(f"Distribuicao de classes (coluna '{label_col}'):")
    print(labels_df[label_col].value_counts().to_string())
    
    # verifica imagens existentes
    img_files = {p.name for p in img_dir.iterdir() if p.is_file()}
    label_files = set(labels_df[img_col].astype(str))
    
    missing_imgs = label_files - img_files
    extra_imgs = img_files - label_files
    print(f"\nArquivos na pasta de imagens: {len(img_files)}")
    print(f"Linhas no rotulo sem imagem correspondente: {len(missing_imgs)}")
    print(f"Imagens sem rotulo correspondente: {len(extra_imgs)}")
    
    # analise de resolucao por classe
    corrupted = []
    sizes_by_class = {0: [], 1: []}
    
    for _, row in labels_df.iterrows():
        img_name = str(row[img_col])
        label = int(row[label_col])
        img_path = img_dir / img_name
        if not img_path.exists():
            continue
        try:
            with Image.open(img_path) as im:
                sizes_by_class[label].append(im.size)
        except Exception as e:
            corrupted.append((str(img_path), str(e)))
    
    print(f"\nArquivos corrompidos: {len(corrupted)}")
    if corrupted:
        for path, err in corrupted[:5]:
            print(f"  {path}: {err}")
    
    for label, sizes in sizes_by_class.items():
        if not sizes:
            continue
        label_name = "Healthy (normal)" if label == 1 else "Unhealthy (PCOS)"
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        print(f"\n[{label_name}] n={len(sizes)}")
        print(f"  Largura: min={min(widths)}, max={max(widths)}, media={sum(widths)//len(widths)}")
        print(f"  Altura : min={min(heights)}, max={max(heights)}, media={sum(heights)//len(heights)}")
    
    return sizes_by_class

# --- TRAIN ---
train_df = load_labels(TRAIN_LABELS)
print(f"\nColunas do train: {list(train_df.columns)}")
# detecta nome correto da coluna de imagem
img_col_train = "imagePath" if "imagePath" in train_df.columns else train_df.columns[0]
label_col_train = "Healthy" if "Healthy" in train_df.columns else train_df.columns[1]
analyze_split("train", TRAIN_IMG_DIR, train_df, label_col_train, img_col_train)

# --- TEST ---
test_df = load_labels(TEST_LABELS)
print(f"\nColunas do test: {list(test_df.columns)}")
img_col_test = "imagePath" if "imagePath" in test_df.columns else test_df.columns[0]
label_col_test = "Healthy" if "Healthy" in test_df.columns else test_df.columns[1]
analyze_split("test", TEST_IMG_DIR, test_df, label_col_test, img_col_test)

# --- amostras visuais do train ---
print("\nGerando figura com amostras do train...")
fig, axes = plt.subplots(2, SAMPLES_PER_CLASS, figsize=(SAMPLES_PER_CLASS * 2, 5))
fig.suptitle("Amostras do dataset PCOSGen (train)", fontsize=14)

random.seed(42)
class_names = {1: "Healthy (normal)", 0: "Unhealthy (PCOS)"}
for row, label in enumerate([1, 0]):  # primeiro normal, depois PCOS
    samples_df = train_df[train_df[label_col_train] == label].sample(
        n=SAMPLES_PER_CLASS, random_state=42
    )
    for col, (_, sample_row) in enumerate(samples_df.iterrows()):
        img_name = str(sample_row[img_col_train])
        img_path = TRAIN_IMG_DIR / img_name
        if img_path.exists():
            with Image.open(img_path) as im:
                axes[row, col].imshow(im, cmap="gray")
        axes[row, col].axis("off")
        if col == 0:
            axes[row, col].set_ylabel(class_names[label], fontsize=10)

plt.tight_layout()
output_path = OUTPUT_DIR / "03_pcosgen_samples.png"
plt.savefig(output_path, dpi=100, bbox_inches="tight")
print(f"Figura salva em: {output_path}")
plt.close()

print("\nSanity check concluido.\n")