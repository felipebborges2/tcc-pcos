"""
gerar_resumo_ablacao.py
Figura-resumo da ablacao (rodar_ablacao.py), a partir das metricas e das
predicoes por imagem salvas em cada passo:
- painel A: AUC por passo (PCOSGen-test sem duplicatas, Kaggle, Figshare)
- painel B: especificidade nos externos por passo, com o limiar de Youden
  usado em cada passo e, so' como diagnostico, com limiar fixo 0,5

Uso:
    python src/gerar_resumo_ablacao.py --root outputs/experimentos/ablacao_<data_hora>

Saidas em <root>: resumo_ablacao.png e resumo_ablacao.csv
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROTULOS = ["0 · PR #1", "1 · + limiar\nYouden", "2 · + divisão\npor grupo", "3 · + cinza e\nPadToSquare",
           "4 · + aug.\nultrassom", "5 · + AdamW\ngradual", "6 · + label\nsmoothing"]
SERIES = [  # (dataset em evaluation_results, arquivo de scores, rotulo, cor, marcador)
    ("PCOSGen-test (interno) sem duplicatas", "pcosgen_test", "PCOSGen-test (sem duplicatas)", "#2a78d6", "o"),
    ("Kaggle (externo 1)", "kaggle", "Kaggle", "#eb6834", "s"),
    ("Figshare (externo 2)", "figshare", "Figshare", "#1baf7a", "D"),
]
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


def fmt(x, casas=3):
    return f"{x:.{casas}f}".replace(".", ",")


def estilo(ax):
    ax.grid(axis="y", color=GRADE, linewidth=0.8, zorder=0)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--root", type=Path, required=True)
    root = p.parse_args().root.resolve()
    passos = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("passo"))

    linhas = []
    for i, d in enumerate(passos):
        res = pd.read_csv(d / "avaliacao" / "evaluation_results.csv").set_index("dataset")
        for nome, arq, rot, _, _ in SERIES:
            r = res.loc[nome]
            s = pd.read_csv(d / "avaliacao" / "scores" / f"scores_{arq}.csv")
            normais = s[s["label"] == 0]
            linhas.append({"passo": i, "pasta": d.name, "conjunto": rot, "n": int(r["n_samples"]),
                           "auc": r["auc_roc"], "limiar": r["threshold"],
                           "especificidade": r["specificity"],
                           "especificidade_limiar_0.5": (normais["prob_pcos"] <= 0.5).mean()})
    df = pd.DataFrame(linhas)
    df.to_csv(root / "resumo_ablacao.csv", index=False)

    x = list(range(len(passos)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2))
    for _, _, rot, cor, mk in SERIES:
        d = df[df["conjunto"] == rot]
        ax1.plot(d["passo"], d["auc"], color=cor, marker=mk, linewidth=2, markersize=8,
                 markeredgecolor="white", markeredgewidth=1.5, label=rot, zorder=3)
        ax1.annotate(fmt(d["auc"].iloc[-1]), (d["passo"].iloc[-1], d["auc"].iloc[-1]),
                     xytext=(10, 0), textcoords="offset points", va="center", fontsize=9,
                     color=TINTA_SECUNDARIA)
        ax1.annotate(fmt(d["auc"].iloc[0]), (0, d["auc"].iloc[0]), xytext=(-10, 0),
                     textcoords="offset points", va="center", ha="right", fontsize=9,
                     color=TINTA_SECUNDARIA)
    ax1.set_title("A · AUC-ROC por passo (não depende do limiar)")
    ax1.set_ylabel("AUC-ROC")
    ax1.set_ylim(0.6, 1.0)
    ax1.set_xlim(-0.8, len(passos) - 0.2)

    for _, _, rot, cor, mk in SERIES[1:]:
        d = df[df["conjunto"] == rot]
        ax2.plot(d["passo"], d["especificidade"], color=cor, marker=mk, linewidth=2, markersize=8,
                 markeredgecolor="white", markeredgewidth=1.5, label=f"{rot} · limiar de Youden (usado)",
                 zorder=3)
        ax2.plot(d["passo"], d["especificidade_limiar_0.5"], color=cor, marker=mk, linewidth=1.5,
                 linestyle="--", markersize=6, markerfacecolor="white", markeredgecolor=cor,
                 label=f"{rot} · limiar 0,5 (diagnóstico)", zorder=2)
    lim = df.drop_duplicates("passo")["limiar"].tolist()
    for i, v in enumerate(lim):
        ax2.text(i, 0.02, f"limiar\n{fmt(v)}", ha="center", va="bottom", fontsize=8, color=TINTA_SECUNDARIA)
    ax2.set_title("B · Especificidade nos externos (acerto nas imagens normais)")
    ax2.set_ylabel("Especificidade")
    ax2.set_ylim(0, 1.0)
    ax2.set_xlim(-0.6, len(passos) - 0.4)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor=GRADE, fontsize=8.5)

    for ax in (ax1, ax2):
        ax.set_xticks(x, ROTULOS[:len(passos)], fontsize=8.5)
        ax.set_xlabel("Passo da ablação (cada um acumula os anteriores)")
        estilo(ax)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", edgecolor=GRADE, fontsize=9)
    fig.suptitle("Ablação do relatório técnico — uma execução por passo, sem teste de significância",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(root / "resumo_ablacao.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Salvo em {root / 'resumo_ablacao.png'}")


if __name__ == "__main__":
    main()
