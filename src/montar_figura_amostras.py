from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# ======================= CONFIG =======================
BASE = Path(r"G:\tcc\data")
DATASET = "pcosgen_train"      # "pcosgen_train" | "figshare" | "kaggle"
N_POR_CLASSE = 3               # quantas amostras por classe
SEED = 42
SAIDA = Path(r"G:\tcc") / "figuras"   # onde salvar a figura
CURADAS = {"Normal": ["image0237.jpg", "image0238.jpg", "image2893.jpg"], "PCOS": ["image2633.jpg", "image2662.jpg", "image2677.jpg"]}  

# Caminhos por dataset
FIGSHARE_DIR     = BASE / "figshare" / "PCOS"
KAGGLE_DIR       = BASE / "kaggle" / "data"
PCOSGEN_TRAIN_IMG = BASE / "pcosgen" / "train" / "PCOSGen-train" / "PCOSGen-train" / "images"
PCOSGEN_TRAIN_XL  = BASE / "pcosgen" / "train" / "PCOSGen-train" / "PCOSGen-train" / "class_label.xlsx"

# Colunas do arquivo de rotulos do PCOSGen (ajuste se o script reclamar)
PCOSGEN_LABEL_COL = "Healthy"     # 1 = normal, 0 = PCOS
PCOSGEN_IMG_COL   = "imagePath"   # nome do arquivo da imagem

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# ======================================================


def imgs_em(pasta: Path):
    if not pasta.exists():
        return []
    return [p for p in pasta.iterdir() if p.suffix.lower() in IMG_EXTS]


def eh_normal(nome: str) -> bool:
    n = nome.lower()
    return ("noninf" in n) or ("notinf" in n) or n.startswith("non") or n.startswith("not")


def coletar_por_pastas(raiz: Path):
    """Retorna dict {'Normal': [paths], 'PCOS': [paths]} a partir de pastas de classe."""
    normais, pcos = [], []
    for pasta in raiz.rglob("*"):
        if pasta.is_dir() and ("inf" in pasta.name.lower()):
            (normais if eh_normal(pasta.name) else pcos).extend(imgs_em(pasta))
    return {"Normal": normais, "PCOS": pcos}


def coletar_pcosgen():
    df = pd.read_excel(PCOSGEN_TRAIN_XL)
    for col in (PCOSGEN_LABEL_COL, PCOSGEN_IMG_COL):
        if col not in df.columns:
            raise SystemExit(
                f"Coluna '{col}' nao encontrada. Colunas disponiveis: {list(df.columns)}.\n"
                f"Ajuste PCOSGEN_LABEL_COL / PCOSGEN_IMG_COL na CONFIG."
            )
    normais = [PCOSGEN_TRAIN_IMG / n for n in df.loc[df[PCOSGEN_LABEL_COL] == 1, PCOSGEN_IMG_COL]]
    pcos    = [PCOSGEN_TRAIN_IMG / n for n in df.loc[df[PCOSGEN_LABEL_COL] == 0, PCOSGEN_IMG_COL]]
    return {"Normal": normais, "PCOS": pcos}

def letterbox(img, size):
    """Centraliza a imagem num quadrado de lado `size` com borda preta, sem distorcer."""
    img = img.copy()
    img.thumbnail((size, size))
    canvas = Image.new("L", (size, size), color=0)
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas

def main():
    random.seed(SEED)

    if DATASET == "figshare":
        grupos = coletar_por_pastas(FIGSHARE_DIR)
    elif DATASET == "kaggle":
        grupos = coletar_por_pastas(KAGGLE_DIR)
    elif DATASET == "pcosgen_train":
        grupos = coletar_pcosgen()
    else:
        raise SystemExit(f"DATASET invalido: {DATASET}")

    # seleciona N amostras existentes por classe
   # seleciona N amostras existentes por classe
    selecao = {}
    todos = {p.name: c for c, ps in grupos.items() for p in ps}  # nome -> classe real
    for classe, paths in grupos.items():
        base = {p.name: p for p in paths}
        if CURADAS.get(classe):
            selecao[classe] = [base[n] for n in CURADAS[classe] if n in base]
            for n in CURADAS[classe]:
                if n not in base:
                    if n in todos:
                        print(f"  [aviso] '{n}' nao e da classe {classe}, e sim {todos[n]}.")
                    else:
                        print(f"  [aviso] '{n}' nao existe no dataset (nome errado?).")
        else:
            validos = [p for p in paths if p.exists()]
            selecao[classe] = random.sample(validos, min(N_POR_CLASSE, len(validos)))
        print(f"{classe}: " + ", ".join(p.name for p in selecao[classe]))

    # monta a grade: 2 linhas (Normal, PCOS) x N colunas
    classes = ["Normal", "PCOS"]
    fig, axes = plt.subplots(2, N_POR_CLASSE, figsize=(2.0 * N_POR_CLASSE, 4.4))
    if N_POR_CLASSE == 1:
        axes = axes.reshape(2, 1)

    for i, classe in enumerate(classes):
        for j in range(N_POR_CLASSE):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(selecao[classe]):
                img = Image.open(selecao[classe][j]).convert("L")  # escala de cinzas
                img = letterbox(img, 320)  # quadrado uniforme, borda preta, sem distorcer
                ax.imshow(img, cmap="gray")
            if j == 0:
                ax.set_ylabel(classe, rotation=90, fontsize=11, labelpad=8)
                ax.axis("on")
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

    plt.tight_layout(pad=0.4)
    SAIDA.mkdir(parents=True, exist_ok=True)
    pdf = SAIDA / f"amostras_{DATASET}.pdf"
    png = SAIDA / f"amostras_{DATASET}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=200)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    print(f"Figura salva em:\n  {pdf}\n  {png}")


if __name__ == "__main__":
    main()