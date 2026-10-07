"""
gerar_roc.py
Gera as curvas ROC dos tres conjuntos de teste numa unica figura, a partir dos
scores por imagem salvos por src/evaluate.py em outputs/scores/. Nao precisa
dos dados nem do modelo.

Saida (nesta mesma pasta): curvas_roc.png

Dependencias: matplotlib, numpy.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SAIDA = Path(__file__).resolve().parent
SCORES_DIR = SAIDA.parent / "outputs" / "scores"

# conjunto -> (arquivo de scores, AUC reportada no artigo)
CONJUNTOS = {
    "PCOSGen-test (interno)": ("scores_pcosgen_test.csv", 0.960),
    "Kaggle (externo)": ("scores_kaggle.csv", 0.716),
    "Figshare (externo)": ("scores_figshare.csv", 0.680),
}
CORES = ["#2a78d6", "#eb6834", "#1baf7a"]
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
    return f"{x:.{casas}f}".replace(".", ",")


def ler_scores(arquivo):
    with open(SCORES_DIR / arquivo, newline="") as f:
        linhas = list(csv.DictReader(f))
    labels = np.array([int(l["label"]) for l in linhas])
    logits = np.array([float(l["logit"]) for l in linhas])
    return labels, logits


def curva_roc(labels, scores):
    """Pontos (FPR, TPR) para todos os limiares distintos e a AUC (trapezios)."""
    ordem = np.argsort(-scores, kind="mergesort")
    scores, labels = scores[ordem], labels[ordem]
    # ultimo indice de cada valor distinto de score (empates viram um unico ponto)
    fim = np.r_[np.where(np.diff(scores))[0], len(scores) - 1]
    tp = np.cumsum(labels)[fim]
    fp = (fim + 1) - tp
    tpr = np.r_[0, tp / labels.sum()]
    fpr = np.r_[0, fp / (len(labels) - labels.sum())]
    return fpr, tpr, np.trapezoid(tpr, fpr)


def main():
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.plot([0, 1], [0, 1], color=EIXO, linestyle="--", linewidth=1, zorder=1)

    for (nome, (arquivo, auc_artigo)), cor in zip(CONJUNTOS.items(), CORES):
        labels, logits = ler_scores(arquivo)
        fpr, tpr, auc = curva_roc(labels, logits)
        if round(auc, 3) != auc_artigo:
            raise ValueError(f"{nome}: AUC calculada {auc:.4f} != {auc_artigo}")

        ax.plot(fpr, tpr, color=cor, linewidth=2, zorder=3,
                label=f"{nome}  AUC = {fmt(auc)}")

        # ponto de operacao no limiar 0,5 (logit > 0)
        pred = logits > 0
        sens = (pred & (labels == 1)).sum() / (labels == 1).sum()
        espec = (~pred & (labels == 0)).sum() / (labels == 0).sum()
        ax.plot(1 - espec, sens, "o", markersize=8, color=cor,
                markeredgecolor="white", markeredgewidth=2, zorder=4)

    ax.plot([], [], "o", color=TINTA_SECUNDARIA, markeredgecolor="white",
            markersize=8, label="Limiar 0,5")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ticks = np.arange(0, 1.01, 0.2)
    ax.set_xticks(ticks, [fmt(t, 1) for t in ticks])
    ax.set_yticks(ticks, [fmt(t, 1) for t in ticks])
    ax.set_xlabel("Taxa de falsos positivos (1 − especificidade)")
    ax.set_ylabel("Taxa de verdadeiros positivos (sensibilidade)")
    ax.set_aspect("equal")
    ax.grid(color=GRADE, linewidth=0.8, zorder=0)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.legend(loc="lower right", frameon=True, facecolor="white",
              edgecolor=GRADE, fontsize=9)

    caminho = SAIDA / "curvas_roc.png"
    fig.savefig(caminho, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figura salva em: {caminho}")


if __name__ == "__main__":
    main()
