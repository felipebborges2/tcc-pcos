"""
02_explore_kaggle.py
Sanity check do dataset Kaggle (Choudhari).
- Conta imagens por classe em train e test
- Verifica integridade
- Coleta resolucoes
- Salva amostras visuais
"""

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

# --- configuracao ---
DATA_DIR = Path("G:/tcc/data/kaggle/data")
OUTPUT_DIR = Path("G:/tcc/outputs/figures")
SPLITS = ["train", "test"]
CLASSES = {"notinfected": 0, "infected": 1}  # 0 = normal, 1 = PCOS
SAMPLES_PER_CLASS = 8

# --- estatisticas ---
print("=" * 60)
print("SANITY CHECK - Kaggle PCOS Dataset (Choudhari)")
print("=" * 60)

corrupted = []

for split in SPLITS:
    print(f"\n--- SPLIT: {split} ---")
    for class_name in CLASSES:
        folder = DATA_DIR / split / class_name
        images = list(folder.glob("*"))
        print(f"\n[{split}/{class_name}] {len(images)} arquivos encontrados")

        sizes = []
        formats = []
        for img_path in images:
            try:
                with Image.open(img_path) as im:
                    sizes.append(im.size)
                    formats.append(im.format)
            except Exception as e:
                corrupted.append((str(img_path), str(e)))

        if sizes:
            widths = [s[0] for s in sizes]
            heights = [s[1] for s in sizes]
            print(f"  Largura: min={min(widths)}, max={max(widths)}, "
                  f"media={sum(widths)//len(widths)}")
            print(f"  Altura : min={min(heights)}, max={max(heights)}, "
                  f"media={sum(heights)//len(heights)}")
            print(f"  Formatos: {set(formats)}")

print("\n" + "=" * 60)
print(f"ARQUIVOS CORROMPIDOS: {len(corrupted)}")
print("=" * 60)
if corrupted:
    for path, err in corrupted[:10]:
        print(f"  {path}: {err}")
else:
    print("Nenhum arquivo corrompido encontrado.")

# --- visualizacao de amostras (apenas train) ---
print("\nGerando figura com amostras do train...")

fig, axes = plt.subplots(2, SAMPLES_PER_CLASS, figsize=(SAMPLES_PER_CLASS * 2, 5))
fig.suptitle("Amostras do dataset Kaggle (train)", fontsize=14)

random.seed(42)
for row, class_name in enumerate(CLASSES):
    folder = DATA_DIR / "train" / class_name
    images = list(folder.glob("*"))
    samples = random.sample(images, SAMPLES_PER_CLASS)

    for col, img_path in enumerate(samples):
        with Image.open(img_path) as im:
            axes[row, col].imshow(im, cmap="gray")
        axes[row, col].axis("off")
        if col == 0:
            axes[row, col].set_ylabel(class_name, fontsize=12)

plt.tight_layout()
output_path = OUTPUT_DIR / "02_kaggle_samples.png"
plt.savefig(output_path, dpi=100, bbox_inches="tight")
print(f"Figura salva em: {output_path}")
plt.close()

print("\nSanity check concluido.\n")