"""
gerar_figuras_experimento.py
Figuras de UM experimento, sem comparacao com outros resultados:
- curvas_treino.png: loss, AUC, acuracia, F1 e learning rate por epoca
  (treino x validacao), com a epoca do checkpoint escolhido marcada
- matrizes_confusao.png: matrizes dos tres conjuntos, todas as imagens
- matrizes_confusao_sem_duplicatas.png: idem, sem as duplicatas do PCOSGen-train
- metricas.csv: copia enxuta de evaluation_results.csv

Uso:
    python src/gerar_figuras_experimento.py --exp <pasta> [--log <train_log.csv>] [--titulo "..."]

--log serve para passos sem treino novo (ex.: so' recalibracao do limiar),
que reaproveitam o log do treino original.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASSES = ["Normal (0)", "PCOS (1)"]
COR_TREINO = "#2a78d6"
COR_VAL = "#eb6834"
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
    "axes.titlesize": 11,
})

CONJUNTOS = ["PCOSGen-test (interno)", "Kaggle (externo 1)", "Figshare (externo 2)"]


def fmt(x, casas=3):
    return f"{x:.{casas}f}".replace(".", ",")


def estilo(ax):
    ax.grid(color=GRADE, linewidth=0.8, zorder=0)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


def fig_curvas(log, melhor_epoca, titulo, saida):
    paineis = [("loss", "Loss (BCE ponderada)"), ("auc", "AUC-ROC"),
               ("acc", "Acurácia (limiar 0,5)"), ("f1", "F1-score (limiar 0,5)")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (chave, nome) in zip(axes.flat, paineis):
        ax.plot(log["epoch"], log[f"train_{chave}"], color=COR_TREINO, linewidth=2, label="Treino")
        ax.plot(log["epoch"], log[f"val_{chave}"], color=COR_VAL, linewidth=2, label="Validação")
        b = log[log["epoch"] == melhor_epoca]
        ax.plot(b["epoch"], b[f"val_{chave}"], "o", color=COR_VAL, markersize=9,
                markeredgecolor="white", markeredgewidth=2, zorder=5)
        ax.axvline(melhor_epoca, color=EIXO, linewidth=1, linestyle=":", zorder=1)
        ax.set_title(nome)
        ax.set_xlabel("Época")
        estilo(ax)
    ax = axes.flat[4]
    ax.step(log["epoch"], log["lr"], where="mid", color=TINTA_SECUNDARIA, linewidth=2, label="Backbone")
    if "lr_head" in log and not np.allclose(log["lr_head"], log["lr"]):
        ax.step(log["epoch"], log["lr_head"], where="mid", color=TINTA_SECUNDARIA, linewidth=2,
                linestyle="--", label="Cabeça")
        ax.legend(frameon=False, fontsize=9)
    ax.set_yscale("log")
    ax.set_title("Learning rate (após o scheduler)")
    ax.set_xlabel("Época")
    estilo(ax)
    ax = axes.flat[5]
    ax.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="o", color=COR_VAL, linestyle="", markersize=9,
                        markeredgecolor="white"))
    l.append(f"Checkpoint escolhido (época {melhor_epoca})")
    ax.legend(h, l, loc="upper left", frameon=False, fontsize=10)
    b = log[log["epoch"] == melhor_epoca].iloc[0]
    ax.text(0, 0.45, f"Épocas executadas: {len(log)}\nMelhor época: {melhor_epoca}\n"
                     f"AUC val.: {fmt(b['val_auc'], 4)}\nLoss val.: {fmt(b['val_loss'], 4)}",
            transform=ax.transAxes, fontsize=10, va="top", color=TINTA_SECUNDARIA)
    fig.suptitle(titulo, fontsize=13)
    fig.tight_layout()
    fig.savefig(saida, dpi=200, facecolor="white")
    plt.close(fig)


def fig_matrizes(res, titulo, saida):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    for ax, (_, r) in zip(axes, res.iterrows()):
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{fmt(norm[i, j] * 100, 1)}%\n({cm[i, j]})", ha="center", va="center",
                        fontsize=12, color="white" if norm[i, j] > 0.6 else TINTA)
        ax.set_xticks([0, 1], CLASSES)
        ax.set_yticks([0, 1], CLASSES)
        ax.set_xlabel("Classe prevista")
        ax.set_ylabel("Classe verdadeira")
        ax.set_title(f"{r['dataset']}\nn = {int(r['n_samples'])} · AUC = {fmt(r['auc_roc'])} · "
                     f"sens. {fmt(r['sensitivity'])} · espec. {fmt(r['specificity'])}", fontsize=10)
    fig.suptitle(titulo, fontsize=12)
    fig.tight_layout()
    fig.savefig(saida, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Figuras de um experimento (sem comparacao).")
    p.add_argument("--exp", type=Path, required=True)
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--titulo", default=None)
    args = p.parse_args()

    exp = args.exp.resolve()
    out = exp / "figuras"
    out.mkdir(exist_ok=True)
    titulo = args.titulo or exp.name
    info = json.loads((exp / "avaliacao" / "avaliacao_info.json").read_text(encoding="utf-8"))
    limiar = info["threshold"]

    log_path = args.log or exp / "logs" / "train_log.csv"
    log = pd.read_csv(log_path)
    fig_curvas(log, info["checkpoint_epoch"],
               f"{titulo} — curvas de treino e validação"
               + ("" if args.log is None else f"\n(sem treino novo: log de {log_path.parent.parent.name})"),
               out / "curvas_treino.png")

    res = pd.read_csv(exp / "avaliacao" / "evaluation_results.csv")
    todas = res[res["subconjunto"] == "todas"].set_index("dataset").loc[CONJUNTOS].reset_index()
    fig_matrizes(todas, f"{titulo} — matrizes de confusão, todas as imagens "
                        f"(limiar {fmt(limiar, 4)}; 0 = Normal, 1 = PCOS; % por classe verdadeira)",
                 out / "matrizes_confusao.png")
    sem = res[res["subconjunto"] != "todas"]
    if len(sem):
        fig_matrizes(sem.reset_index(drop=True),
                     f"{titulo} — matrizes de confusão sem as duplicatas do PCOSGen-train "
                     f"(limiar {fmt(limiar, 4)})", out / "matrizes_confusao_sem_duplicatas.png")
    cols = ["dataset", "subconjunto", "n_samples", "n_normal", "n_pcos", "accuracy", "precision",
            "sensitivity", "specificity", "f1_score", "auc_roc", "tn", "fp", "fn", "tp", "threshold"]
    res[[c for c in cols if c in res]].to_csv(out / "metricas.csv", index=False)
    print(f"Figuras salvas em {out}")


if __name__ == "__main__":
    main()
