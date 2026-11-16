"""
gerar_figuras_dedup_cv.py
Figuras da validacao cruzada na base deduplicada (treinar_dedup_cv.py):
- curvas_treino_validacao.png: loss e AUC de treino x validacao por epoca,
  cada fold em linha fina e a media entre folds (com faixa de +-1 desvio) em
  linha grossa; ponto na epoca escolhida de cada fold
- curvas_acuracia_f1_lr.png: acuracia e F1 de treino x validacao por epoca
  (mesmo formato) e a taxa de aprendizado de cada fold
- roc_externos.png: curvas ROC no Kaggle e no Figshare, uma por fold
- metricas_resumo.png: media +- desvio entre folds (validacao, Kaggle, Figshare)

Uso:
    python src/gerar_figuras_dedup_cv.py --exp outputs/experimentos/dedup_pelvinext_cv_<data_hora>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

COR_TREINO = "#2a78d6"
COR_VAL = "#eb6834"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": EIXO, "axes.labelcolor": TINTA_SECUNDARIA,
    "xtick.color": TINTA_SECUNDARIA, "ytick.color": TINTA_SECUNDARIA, "text.color": TINTA,
    "axes.titlesize": 11,
})


def fmt(x, casas=3):
    return f"{x:.{casas}f}".replace(".", ",")


def estilo(ax):
    ax.grid(color=GRADE, linewidth=0.8, zorder=0)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


def _ler_logs(exp, folds):
    logs = {k: pd.read_csv(exp / f"fold{k}" / "logs" / "train_log.csv") for k in folds}
    cfgs = {k: json.loads((exp / f"fold{k}" / "config.json").read_text(encoding="utf-8")) for k in folds}
    return logs, cfgs


def _painel_treino_val(ax, logs, cfgs, folds, chave, titulo):
    for lado, cor, rot in [("train", COR_TREINO, "Treino"), ("val", COR_VAL, "Validação")]:
        tab = pd.concat([l.set_index("epoch")[f"{lado}_{chave}"].rename(k) for k, l in logs.items()], axis=1)
        for k in folds:
            ax.plot(tab.index, tab[k], color=cor, linewidth=0.8, alpha=0.35, zorder=2)
        media, dp, n = tab.mean(axis=1), tab.std(axis=1), tab.count(axis=1)
        media, dp = media[n >= 3], dp[n >= 3]  # media so' onde ha pelo menos 3 folds
        ax.fill_between(media.index, media - dp, media + dp, color=cor, alpha=0.12, linewidth=0, zorder=1)
        ax.plot(media.index, media, color=cor, linewidth=2.4, label=f"{rot} — média dos folds (±1 desvio)",
                zorder=3)
        if lado == "val":
            for k in folds:
                e = cfgs[k]["melhor_epoca"]
                ax.plot(e, tab.loc[e, k], "o", color=cor, markersize=7, markeredgecolor="white",
                        markeredgewidth=1.5, zorder=4)
    ax.set_title(titulo)
    ax.set_xlabel("Época")
    estilo(ax)


def _legenda_e_nota(fig, ax_ref, logs, y_leg=0.03):
    h, l = ax_ref.get_legend_handles_labels()
    h += [plt.Line2D([], [], color=TINTA_SECUNDARIA, linewidth=0.8, alpha=0.5),
          plt.Line2D([], [], marker="o", color=COR_VAL, linestyle="", markersize=7, markeredgecolor="white")]
    l += ["Cada fold (linha fina)", "Época escolhida em cada fold (maior AUC de validação)"]
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, y_leg))
    if all(0 in lg["epoch"].values for lg in logs.values()):
        fig.text(0.5, 0.005, "Época 0 = modelo antes do treino (treino avaliado sem augmentation); "
                 "nas demais épocas, o valor de treino é a média durante a época, com augmentation.",
                 ha="center", fontsize=8.5, color=TINTA_SECUNDARIA)


def _titulo(cfgs, folds, prefixo):
    n_tr = cfgs[folds[0]]["n_treino"]; n_va = cfgs[folds[0]]["n_validacao"]
    return (f"{prefixo} — PCOSGen deduplicada (225 imagens), validação cruzada em {len(folds)} folds "
            f"(~{n_tr} treino / ~{n_va} validação por fold)")


def fig_curvas(exp, folds, saida):
    logs, cfgs = _ler_logs(exp, folds)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    _painel_treino_val(axes[0], logs, cfgs, folds, "loss", "Loss (BCE ponderada)")
    _painel_treino_val(axes[1], logs, cfgs, folds, "auc", "AUC-ROC")
    _legenda_e_nota(fig, axes[0], logs)
    fig.suptitle(_titulo(cfgs, folds, "Treino × validação"), fontsize=12)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(saida, dpi=200, facecolor="white")
    plt.close(fig)


def fig_outras_metricas(exp, folds, saida):
    """Acuracia e F1 (limiar 0,5) de treino x validacao por epoca e a taxa de aprendizado."""
    logs, cfgs = _ler_logs(exp, folds)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    _painel_treino_val(axes[0], logs, cfgs, folds, "acc", "Acurácia (limiar 0,5)")
    _painel_treino_val(axes[1], logs, cfgs, folds, "f1", "F1 (limiar 0,5)")
    ax = axes[2]
    cores_fold = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#c9a227"]
    for i, k in enumerate(folds):
        lg = logs[k]
        ax.step(lg["epoch"], lg["lr"], where="post", color=cores_fold[i % len(cores_fold)], linewidth=1.6,
                alpha=0.85, zorder=2 + i, label=f"Fold {k} (parou na época {lg['epoch'].max()})")
    ax.set_yscale("log")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=GRADE, fontsize=8)
    ax.set_title("Taxa de aprendizado (ReduceLROnPlateau)")
    ax.set_xlabel("Época")
    estilo(ax)
    _legenda_e_nota(fig, axes[0], logs)
    fig.suptitle(_titulo(cfgs, folds, "Outras métricas por época"), fontsize=12)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(saida, dpi=200, facecolor="white")
    plt.close(fig)


def fig_roc(exp, folds, saida):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))
    for ax, nome in zip(axes, ["Kaggle", "Figshare"]):
        aucs = []
        for k in folds:
            s = pd.read_csv(exp / f"fold{k}" / f"scores_{nome.lower()}.csv")
            fpr, tpr, _ = roc_curve(s["label"], s["logit"])
            auc = roc_auc_score(s["label"], s["logit"])
            aucs.append(auc)
            ax.plot(fpr, tpr, color=COR_VAL, linewidth=1.4, alpha=0.6,
                    label=f"Fold {k}  AUC = {fmt(auc)}", zorder=3)
        ax.plot([0, 1], [0, 1], color=EIXO, linestyle="--", linewidth=1, zorder=1)
        ax.set_title(f"{nome} (n = {len(s)}) — AUC média {fmt(np.mean(aucs))} ± {fmt(np.std(aucs, ddof=1))}")
        ax.set_xlabel("Taxa de falsos positivos (1 − especificidade)")
        ax.set_ylabel("Sensibilidade")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
        estilo(ax)
        ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=GRADE, fontsize=8)
    fig.suptitle("Curvas ROC nos conjuntos externos — um modelo por fold (treinado na base deduplicada)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(saida, dpi=200, facecolor="white")
    plt.close(fig)


def fig_resumo(exp, saida):
    r = pd.read_csv(exp / "metricas_por_fold.csv")
    r = r[(r["subconjunto"] == "todas") & (r["limiar_tipo"] == "0,5")]
    nomes = {"Validacao do fold (interno)": "Validação (interno)", "Kaggle": "Kaggle", "Figshare": "Figshare"}
    metricas = [("auc_roc", "AUC"), ("accuracy", "Acurácia"), ("sensitivity", "Sensibilidade"),
                ("specificity", "Especificidade"), ("f1_score", "F1")]
    cores = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(metricas))
    larg = 0.26
    for i, (ds, rot) in enumerate(nomes.items()):
        d = r[r["dataset"] == ds]
        m = [d[c].mean() for c, _ in metricas]
        s = [d[c].std() for c, _ in metricas]
        pos = x + (i - 1) * (larg + 0.01)
        ax.bar(pos, m, width=larg, color=cores[i], label=rot, zorder=3)
        ax.errorbar(pos, m, yerr=s, fmt="none", ecolor=TINTA_SECUNDARIA, capsize=3, linewidth=1, zorder=4)
        for p, v in zip(pos, m):
            ax.text(p, 0.02, fmt(v, 2), ha="center", va="bottom", fontsize=8, color="white", zorder=5)
    ax.set_xticks(x, [n for _, n in metricas])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=GRADE, linewidth=0.8, zorder=0)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=GRADE, fontsize=9)
    ax.set_title("Média ± desvio entre os 5 folds (limiar 0,5) — interno = validação de cada fold")
    fig.tight_layout()
    fig.savefig(saida, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--exp", type=Path, required=True)
    exp = p.parse_args().exp.resolve()
    folds = sorted(int(d.name[4:]) for d in exp.iterdir() if d.is_dir() and d.name.startswith("fold")
                   and (d / "config.json").exists())
    out = exp / "figuras"
    out.mkdir(exist_ok=True)
    fig_curvas(exp, folds, out / "curvas_treino_validacao.png")
    fig_outras_metricas(exp, folds, out / "curvas_acuracia_f1_lr.png")
    fig_roc(exp, folds, out / "roc_externos.png")
    fig_resumo(exp, out / "metricas_resumo.png")
    print(f"Figuras salvas em {out}")


if __name__ == "__main__":
    main()
