"""
01_explore_dataset.py
Sanity check do dataset Figshare PCOS.
- Conta imagens por classe
- Verifica integridade (arquivos corrompidos)
- Coleta resolucoes
- Salva amostras visuais para inspecao
"""

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

# --- configuracao ---
DATA_DIR = Path("G:/tcc/data/figshare/PCOS")
OUTPUT_DIR = Path("G:/tcc/outputs/figures")
CLASSES = {"noninfected": 0, "infected": 1}  # 0 = normal, 1 = PCOS
SAMPLES_PER_CLASS = 8  # quantas imagens visualizar de cada classe

# --- estatisticas ---
print("=" * 60)
print("SANITY CHECK - Figshare PCOS Dataset")
print("=" * 60)

stats = {}
corrupted = []

for class_name in CLASSES:
    folder = DATA_DIR / class_name
    images = list(folder.glob("*"))
    print(f"\n[{class_name}] {len(images)} arquivos encontrados")
    
    sizes = []
    formats = []
    for img_path in images:
        try:
            with Image.open(img_path) as im:
                sizes.append(im.size)
                formats.append(im.format)
        except Exception as e:
            corrupted.append((str(img_path), str(e)))
    
    # estatisticas de resolucao
    if sizes:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        print(f"  Largura: min={min(widths)}, max={max(widths)}, "
              f"media={sum(widths)//len(widths)}")
        print(f"  Altura : min={min(heights)}, max={max(heights)}, "
              f"media={sum(heights)//len(heights)}")
        print(f"  Formatos: {set(formats)}")
    
    stats[class_name] = {
        "count": len(images),
        "sizes": sizes,
        "formats": set(formats),
    }

print("\n" + "=" * 60)
print(f"ARQUIVOS CORROMPIDOS: {len(corrupted)}")
print("=" * 60)
if corrupted:
    for path, err in corrupted[:10]:
        print(f"  {path}: {err}")
else:
    print("Nenhum arquivo corrompido encontrado.")

# --- visualizacao de amostras ---
print("\nGerando figura com amostras...")

fig, axes = plt.subplots(2, SAMPLES_PER_CLASS, figsize=(SAMPLES_PER_CLASS * 2, 5))
fig.suptitle("Amostras do dataset Figshare PCOS", fontsize=14)

random.seed(42)  # reproduzivel
for row, class_name in enumerate(CLASSES):
    folder = DATA_DIR / class_name
    images = list(folder.glob("*"))
    samples = random.sample(images, SAMPLES_PER_CLASS)
    
    for col, img_path in enumerate(samples):
        with Image.open(img_path) as im:
            axes[row, col].imshow(im, cmap="gray")
        axes[row, col].axis("off")
        if col == 0:
            axes[row, col].set_ylabel(class_name, fontsize=12)

plt.tight_layout()
output_path = OUTPUT_DIR / "01_dataset_samples.png"
plt.savefig(output_path, dpi=100, bbox_inches="tight")
print(f"Figura salva em: {output_path}")
plt.close()

print("\nSanity check concluido.\n")