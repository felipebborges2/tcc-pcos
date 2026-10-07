"""
gerar_figuras.py
Gera as figuras de resultados a partir das matrizes de confusao e metricas
reportadas no artigo, declaradas como constantes abaixo. Nao depende dos dados
nem do modelo treinado.

Saidas (nesta mesma pasta):
- matrizes_confusao.png   : matrizes normalizadas por linha (classe verdadeira)
- comparacao_metricas.png : barras agrupadas com as cinco metricas por conjunto

Dependencias: matplotlib, numpy.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SAIDA = Path(__file__).resolve().parent

CONJUNTOS = ["PCOSGen-test (interno)", "Kaggle (externo)", "Figshare (externo)"]

# Matrizes de confusao (contagens absolutas).
# Linhas = classe verdadeira [Normal, PCOS]; colunas = classe predita [Normal, PCOS].
MATRIZES = {
    "PCOSGen-test (interno)": np.array([[339, 78], [49, 1002]]),
    "Kaggle (externo)": np.array([[855, 1433], [390, 1178]]),
    "Figshare (externo)": np.array([[2441, 2559], [1875, 4909]]),
}

METRICAS = ["AUC-ROC", "F1", "Acurácia", "Sensib.", "Especif."]
VALORES = {
    "PCOSGen-test (interno)": [0.960, 0.940, 0.913, 0.953, 0.813],
    "Kaggle (externo)": [0.716, 0.564, 0.527, 0.751, 0.374],
    "Figshare (externo)": [0.680, 0.689, 0.624, 0.724, 0.488],
}

CLASSES = ["Normal", "PCOS"]
CORES = ["#2a78d6", "#eb6834", "#1baf7a"]  # um tom fixo por conjunto
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": EIXO,
    "axes.labelcolor": TINTA_SECUNDARIA,
    "xtick.color": TINTA_SECUNDARIA,
    "ytick.color": TINTA_SECUNDARIA,
    "text.color": TINTA,
})


def fmt(x, casas=3):
    """Formata numero com virgula decimal."""
    return f"{x:.{casas}f}".replace(".", ",")


def conferir_consistencia():
    """Confere se as metricas derivaveis das matrizes batem com a tabela (3 casas)."""
    for nome in CONJUNTOS:
        (tn, fp), (fn, tp) = MATRIZES[nome]
        derivadas = {
            "F1": 2 * tp / (2 * tp + fp + fn),
            "Acurácia": (tp + tn) / (tn + fp + fn + tp),
            "Sensib.": tp / (tp + fn),
            "Especif.": tn / (tn + fp),
        }
        for metrica, valor in derivadas.items():
            declarado = VALORES[nome][METRICAS.index(metrica)]
            if abs(round(valor, 3) - declarado) > 1e-9:
                raise ValueError(
                    f"{nome}: {metrica} derivada da matriz = {valor:.4f}, "
                    f"tabela = {declarado:.3f}"
                )


def figura_matrizes():
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2),
                             gridspec_kw={"wspace": 0.35})
    cmap = plt.get_cmap("Blues")

    for ax, nome in zip(axes, CONJUNTOS):
        cm = MATRIZES[nome]
        norm = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)

        for i in range(2):
            for j in range(2):
                cor = "white" if norm[i, j] > 0.55 else TINTA
                ax.text(j, i, f"{cm[i, j]:,}".replace(",", "."),
                        ha="center", va="center", color=cor,
                        fontsize=13, fontweight="bold")
                ax.text(j, i + 0.2, f"{fmt(norm[i, j] * 100, 1)}%",
                        ha="center", va="center", color=cor, fontsize=10)

        n = cm.sum()
        ax.set_title(f"{nome}\nn = {n:,}".replace(",", "."), fontsize=11)
        ax.set_xticks([0, 1], CLASSES)
        ax.set_yticks([0, 1], CLASSES)
        ax.set_xlabel("Classe predita")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].set_ylabel("Classe verdadeira")
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Proporção por classe verdadeira")
    cbar.outline.set_visible(False)

    caminho = SAIDA / "matrizes_confusao.png"
    fig.savefig(caminho, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return caminho


def figura_metricas():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(METRICAS))
    largura = 0.26

    for k, (nome, cor) in enumerate(zip(CONJUNTOS, CORES)):
        pos = x + (k - 1) * largura
        barras = ax.bar(pos, VALORES[nome], width=largura - 0.02,
                        color=cor, label=nome, zorder=3)
        for b, v in zip(barras, VALORES[nome]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, fmt(v),
                    ha="center", va="bottom", fontsize=7.5,
                    color=TINTA_SECUNDARIA)

    ax.set_xticks(x, METRICAS)
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2),
                  [fmt(v, 1) for v in np.arange(0, 1.01, 0.2)])
    ax.set_ylabel("Valor")
    ax.grid(axis="y", color=GRADE, linewidth=0.8, zorder=0)
    ax.tick_params(axis="x", length=0)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3,
              frameon=False)

    caminho = SAIDA / "comparacao_metricas.png"
    fig.savefig(caminho, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return caminho


def main():
    conferir_consistencia()
    for caminho in (figura_matrizes(), figura_metricas()):
        print(f"Figura salva em: {caminho}")


if __name__ == "__main__":
    main()
