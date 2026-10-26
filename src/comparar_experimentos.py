"""
comparar_experimentos.py
Compara o experimento de referencia (antes) com o experimento do PR #1 do Luan
(depois) nos mesmos conjuntos de avaliacao, com a mesma definicao de classes
(0 = normal, 1 = PCOS, classe positiva) e o mesmo limiar fixo (prob_pcos > 0,5).

Tudo e' calculado a partir das predicoes por imagem (scores) e dos logs de
treino reais. Nada e' reconstruido a partir de metricas agregadas.

Entradas:
- antes: scores historicos em outputs/scores/, metricas historicas em
  outputs/logs/evaluation_results.csv, log em outputs/logs/train_log.csv e
  checkpoint em outputs/models/best_model.pt (so' para ler a epoca salva).
  Opcional: --reavaliacao-antes <pasta>, saida de evaluate.py rodado de novo no
  checkpoint antigo, para conferir se reproduz os scores historicos.
- depois: pasta do experimento (train.py + evaluate.py), com
  models/best_model.pt, logs/train_log.csv e avaliacao/.

Saidas (padrao: <pasta do experimento depois>/comparacao/):
- metricas_por_conjunto.csv, diferencas_antes_depois.csv,
  lacuna_generalizacao.csv, resumo_treino.csv, verificacoes.csv
- predicoes_por_imagem_<conjunto>.csv
- curvas_treino_antes_depois.png, roc_<conjunto>.png,
  matriz_confusao_<conjunto>.png, generalizacao_interno_externo.png
- RELATORIO.md

Uso:
    python src/comparar_experimentos.py --depois <pasta do experimento>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

RAIZ = Path(__file__).resolve().parent.parent
THRESHOLD = 0.5
CLASSES = ["Normal (0)", "PCOS (1)"]  # ordem das linhas/colunas das matrizes

# chave -> (arquivo de scores, nome no evaluation_results.csv, nome curto, tipo)
CONJUNTOS = {
    "pcosgen_test": ("scores_pcosgen_test.csv", "PCOSGen-test (interno)", "PCOSGen-test", "interno"),
    "kaggle": ("scores_kaggle.csv", "Kaggle (externo 1)", "Kaggle", "externo"),
    "figshare": ("scores_figshare.csv", "Figshare (externo 2)", "Figshare", "externo"),
}

# metrica -> (rotulo, e' porcentagem?)
METRICAS = {
    "acuracia": ("Acurácia", True),
    "precisao": ("Precisão", True),
    "sensibilidade": ("Sensibilidade (recall PCOS)", True),
    "especificidade": ("Especificidade", True),
    "f1": ("F1-score", True),
    "auc_roc": ("AUC-ROC", False),
}

COR = {"antes": "#2a78d6", "depois": "#eb6834"}
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


def parse_args():
    p = argparse.ArgumentParser(description="Compara experimento de referencia x experimento do PR #1.")
    p.add_argument("--depois", type=Path, required=True, help="pasta do experimento novo")
    p.add_argument("--antes-scores", type=Path, default=RAIZ / "outputs" / "scores")
    p.add_argument("--antes-resultados", type=Path, default=RAIZ / "outputs" / "logs" / "evaluation_results.csv")
    p.add_argument("--antes-log", type=Path, default=RAIZ / "outputs" / "logs" / "train_log.csv")
    p.add_argument("--antes-checkpoint", type=Path, default=RAIZ / "outputs" / "models" / "best_model.pt")
    p.add_argument("--reavaliacao-antes", type=Path, default=None,
                   help="saida de evaluate.py no checkpoint antigo (opcional, para conferencia)")
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


# ============================================================
# LEITURA E METRICAS
# ============================================================

def ler_scores(caminho):
    df = pd.read_csv(caminho)
    if "pred_label" not in df.columns:
        # scores historicos nao tem a coluna; mesmo criterio de evaluate.py
        df["pred_label"] = (df["prob_pcos"] > THRESHOLD).astype(int)
    return df


def metricas(df):
    y, pred = df["label"].values, df["pred_label"].values
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: a / b if b else float("nan")
    return {
        "n": len(df), "n_normal": int((y == 0).sum()), "n_pcos": int((y == 1).sum()),
        "acuracia": (tp + tn) / len(df),
        "precisao": div(tp, tp + fp),
        "sensibilidade": div(tp, tp + fn),
        "especificidade": div(tn, tn + fp),
        "f1": div(2 * tp, 2 * tp + fp + fn),
        "auc_roc": roc_auc_score(y, df["logit"].values),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "limiar": THRESHOLD,
    }


def melhor_epoca(checkpoint, log):
    epoca, fonte = None, ""
    if checkpoint.exists():
        import torch
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        epoca, fonte = int(ck["epoch"]), "checkpoint"
    # criterio de train.py: primeira epoca com o maior val_auc
    epoca_log = int(log.loc[log["val_auc"].idxmax(), "epoch"])
    if epoca is None:
        epoca, fonte = epoca_log, "log (maior val_auc)"
    return epoca, epoca_log, fonte


# ============================================================
# FIGURAS
# ============================================================

def estilo(ax):
    ax.grid(color=GRADE, linewidth=0.8, zorder=0)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


def fig_curvas_treino(logs, epocas, rotulos, head_only, saida):
    paineis = [("loss", "Loss (BCE ponderada)"), ("acc", "Acurácia"),
               ("auc", "AUC-ROC"), ("f1", "F1-score")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (chave, titulo) in zip(axes.flat, paineis):
        for exp in ("antes", "depois"):
            log = logs[exp]
            ax.plot(log["epoch"], log[f"train_{chave}"], color=COR[exp], linewidth=2,
                    linestyle="--", label=f"{rotulos[exp]} · treino")
            ax.plot(log["epoch"], log[f"val_{chave}"], color=COR[exp], linewidth=2,
                    label=f"{rotulos[exp]} · validação")
            linha = log[log["epoch"] == epocas[exp]]
            ax.plot(linha["epoch"], linha[f"val_{chave}"], "o", color=COR[exp], markersize=9,
                    markeredgecolor="white", markeredgewidth=2, zorder=5)
        ax.axvspan(0.5, head_only + 0.5, color=COR["depois"], alpha=0.08, linewidth=0)
        ax.set_title(titulo)
        ax.set_xlabel("Época")
        estilo(ax)
    ax = axes.flat[4]
    for exp in ("antes", "depois"):
        ax.step(logs[exp]["epoch"], logs[exp]["lr"], where="mid", color=COR[exp],
                linewidth=2, label=rotulos[exp])
    ax.axvspan(0.5, head_only + 0.5, color=COR["depois"], alpha=0.08, linewidth=0)
    ax.set_yscale("log")
    ax.set_title("Learning rate (após o scheduler)")
    ax.set_xlabel("Época")
    estilo(ax)

    ax = axes.flat[5]
    ax.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="o", color=TINTA_SECUNDARIA, linestyle="",
                        markersize=9, markeredgecolor="white"))
    l.append("Época do melhor checkpoint (maior AUC val.)")
    h.append(plt.Rectangle((0, 0), 1, 1, color=COR["depois"], alpha=0.15))
    l.append(f"Depois: épocas 1–{head_only} só com a cabeça treinável")
    ax.legend(h, l, loc="upper left", frameon=False, fontsize=10)
    texto = "\n".join(
        f"{rotulos[e]}: melhor época {epocas[e]} de {len(logs[e])}, "
        f"AUC val. = {fmt(logs[e].loc[logs[e]['epoch'] == epocas[e], 'val_auc'].iloc[0], 4)}"
        for e in ("antes", "depois"))
    ax.text(0, 0.25, texto, transform=ax.transAxes, fontsize=10, va="top", color=TINTA_SECUNDARIA)
    fig.suptitle("Curvas de treino e validação (PCOSGen-train) — antes × depois", fontsize=13)
    fig.tight_layout()
    fig.savefig(saida, dpi=200, facecolor="white")
    plt.close(fig)


def fig_roc(dfs, mets, rotulos, nome, saida):
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.plot([0, 1], [0, 1], color=EIXO, linestyle="--", linewidth=1, zorder=1)
    for exp in ("antes", "depois"):
        df, m = dfs[exp], mets[exp]
        fpr, tpr, _ = roc_curve(df["label"], df["logit"])
        ax.plot(fpr, tpr, color=COR[exp], linewidth=2, zorder=3,
                label=f"{rotulos[exp]}  AUC = {fmt(m['auc_roc'])}")
        ax.plot(1 - m["especificidade"], m["sensibilidade"], "o", markersize=9, color=COR[exp],
                markeredgecolor="white", markeredgewidth=2, zorder=4)
    ax.plot([], [], "o", color=TINTA_SECUNDARIA, markeredgecolor="white", markersize=9,
            label="Ponto de operação no limiar 0,5")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.set_xlabel("Taxa de falsos positivos (1 − especificidade)")
    ax.set_ylabel("Taxa de verdadeiros positivos (sensibilidade)")
    ax.set_title(f"Curva ROC — {nome} (n = {mets['depois']['n']})")
    estilo(ax)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=GRADE, fontsize=9)
    fig.savefig(saida, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_matrizes(mets, rotulos, nome, saida):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.2))
    for col, exp in enumerate(("antes", "depois")):
        m = mets[exp]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        norm = cm / cm.sum(axis=1, keepdims=True)
        for lin, (mat, titulo) in enumerate([(cm, "contagens"), (norm, "normalizada por classe verdadeira")]):
            ax = axes[lin, col]
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    txt = f"{cm[i, j]}" if lin == 0 else f"{fmt(norm[i, j] * 100, 1)}%"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=13,
                            color="white" if norm[i, j] > 0.6 else TINTA)
            ax.set_xticks([0, 1], CLASSES)
            ax.set_yticks([0, 1], CLASSES)
            ax.set_xlabel("Classe prevista")
            ax.set_ylabel("Classe verdadeira")
            ax.set_title(f"{rotulos[exp]} — {titulo}", fontsize=10)
    fig.suptitle(f"Matrizes de confusão — {nome} (n = {mets['depois']['n']}, limiar 0,5)\n"
                 "Ordem das classes: linha/coluna 0 = Normal, 1 = PCOS", fontsize=12)
    fig.tight_layout()
    fig.savefig(saida, dpi=200, facecolor="white")
    plt.close(fig)


def fig_generalizacao(tabela, rotulos, saida):
    nomes = [CONJUNTOS[k][2] for k in CONJUNTOS]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    x = np.arange(len(nomes))
    for ax, (chave, (titulo, _)) in zip(axes.flat, METRICAS.items()):
        for exp, desloc in (("antes", -0.06), ("depois", 0.06)):
            v = [tabela[(exp, k)][chave] for k in CONJUNTOS]
            ax.plot(x + desloc, v, "-o", color=COR[exp], linewidth=2, markersize=8,
                    markeredgecolor="white", markeredgewidth=2, label=rotulos[exp], zorder=3)
            ax.annotate(fmt(v[0]), (x[0] + desloc, v[0]), textcoords="offset points",
                        xytext=(-28 if exp == "antes" else 6, 0), va="center",
                        fontsize=8, color=TINTA_SECUNDARIA)
        ax.axvspan(0.5, 2.5, color=GRADE, alpha=0.35, linewidth=0, zorder=0)
        ax.set_xticks(x, nomes)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, 1.02)
        ax.set_title(titulo)
        estilo(ax)
    axes.flat[0].legend(loc="lower left", frameon=True, facecolor="white", edgecolor=GRADE, fontsize=9)
    fig.suptitle("Desempenho interno (PCOSGen-test) × externo (Kaggle, Figshare; faixa cinza) — limiar 0,5",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(saida, dpi=200, facecolor="white")
    plt.close(fig)


def fig_barras_metricas(tabela, rotulos, saida):
    """Barras agrupadas: as seis metricas antes x depois, um painel por conjunto."""
    chaves = list(METRICAS)
    nomes = [METRICAS[c][0].split(" (")[0] for c in chaves]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharey=True)
    x = np.arange(len(chaves))
    larg = 0.36
    for ax, (k, (_, _, nome, tipo)) in zip(axes, CONJUNTOS.items()):
        for exp, desloc in (("antes", -larg / 2 - 0.01), ("depois", larg / 2 + 0.01)):
            v = [tabela[(exp, k)][c] for c in chaves]
            barras = ax.bar(x + desloc, v, width=larg, color=COR[exp], label=rotulos[exp], zorder=3)
            for b, val in zip(barras, v):
                ax.text(b.get_x() + b.get_width() / 2, val + 0.012, fmt(val, 2),
                        ha="center", va="bottom", fontsize=7.5, color=TINTA_SECUNDARIA)
        ax.set_xticks(x, nomes, rotation=30, ha="right")
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{nome} ({tipo}, n = {tabela[('depois', k)]['n']})")
        ax.grid(axis="y", color=GRADE, linewidth=0.8, zorder=0)
        ax.grid(axis="x", visible=False)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    fig.suptitle("Métricas por conjunto — antes × depois (limiar 0,5; classe positiva PCOS)", fontsize=13)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(saida, dpi=200, facecolor="white")
    plt.close(fig)


def fig_diferencas(tabela, saida):
    """Diferenca depois − antes de cada metrica, por conjunto (p.p.; AUC x100)."""
    chaves = list(METRICAS)
    nomes = [("AUC-ROC (×100)" if c == "auc_roc" else METRICAS[c][0].split(" (")[0]) for c in chaves]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    y = np.arange(len(chaves))[::-1]
    difs_all = [(tabela[("depois", k)][c] - tabela[("antes", k)][c]) * 100
                for k in CONJUNTOS for c in chaves]
    lim = max(abs(min(difs_all)), abs(max(difs_all))) * 1.35 + 0.5
    for ax, (k, (_, _, nome, tipo)) in zip(axes, CONJUNTOS.items()):
        d = [(tabela[("depois", k)][c] - tabela[("antes", k)][c]) * 100 for c in chaves]
        ax.barh(y, d, height=0.55, color=TINTA_SECUNDARIA, zorder=3)
        for yi, val in zip(y, d):
            ax.text(val + (0.15 if val >= 0 else -0.15), yi, f"{val:+.1f}".replace(".", ","),
                    va="center", ha="left" if val >= 0 else "right", fontsize=9, color=TINTA)
        ax.axvline(0, color=TINTA_SECUNDARIA, linewidth=1, zorder=4)
        ax.set_yticks(y, nomes)
        ax.set_xlim(-lim, lim)
        ax.set_title(f"{nome} ({tipo})")
        ax.set_xlabel("depois − antes (pontos percentuais)")
        ax.grid(axis="x", color=GRADE, linewidth=0.8, zorder=0)
        ax.grid(axis="y", visible=False)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    fig.suptitle("Diferença das métricas após o PR #1 (direita = maior depois; uma única execução, "
                 "sem teste de significância)", fontsize=12)
    fig.tight_layout()
    fig.savefig(saida, dpi=200, facecolor="white")
    plt.close(fig)


def fig_matrizes_todas(tabela, rotulos, saida):
    """Matrizes dos tres conjuntos numa figura: linhas = conjuntos, colunas = antes/depois.
    Cor e % normalizados por classe verdadeira; contagem absoluta entre parenteses."""
    fig, axes = plt.subplots(3, 2, figsize=(8.6, 12))
    for lin, (k, (_, _, nome, tipo)) in enumerate(CONJUNTOS.items()):
        for col, exp in enumerate(("antes", "depois")):
            m = tabela[(exp, k)]
            cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
            norm = cm / cm.sum(axis=1, keepdims=True)
            ax = axes[lin, col]
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{fmt(norm[i, j] * 100, 1)}%\n({cm[i, j]})", ha="center",
                            va="center", fontsize=11, color="white" if norm[i, j] > 0.6 else TINTA)
            ax.set_xticks([0, 1], CLASSES)
            ax.set_yticks([0, 1], CLASSES if col == 0 else ["", ""])
            if lin == 2:
                ax.set_xlabel("Classe prevista")
            if col == 0:
                ax.set_ylabel(f"{nome}\nClasse verdadeira")
            ax.set_title(f"{rotulos[exp]} — {nome} (n = {m['n']})", fontsize=10)
    fig.suptitle("Matrizes de confusão — antes × depois (limiar 0,5)\n"
                 "% normalizado por classe verdadeira; contagem entre parênteses; 0 = Normal, 1 = PCOS",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(saida, dpi=200, facecolor="white")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    depois_dir = args.depois.resolve()
    out = (args.out_dir or depois_dir / "comparacao").resolve()
    out.mkdir(parents=True, exist_ok=True)
    rotulos = {"antes": "Antes (referência)", "depois": "Depois (PR #1)"}
    fontes = {
        "antes": {"scores": args.antes_scores, "resultados": args.antes_resultados,
                  "log": args.antes_log, "checkpoint": args.antes_checkpoint},
        "depois": {"scores": depois_dir / "avaliacao" / "scores",
                   "resultados": depois_dir / "avaliacao" / "evaluation_results.csv",
                   "log": depois_dir / "logs" / "train_log.csv",
                   "checkpoint": depois_dir / "models" / "best_model.pt"},
    }
    verif = []

    def check(item, ok, detalhe=""):
        verif.append({"verificacao": item, "ok": bool(ok), "detalhe": detalhe})
        print(f"[{'OK' if ok else 'FALHA'}] {item} {detalhe}")

    # --- predicoes e metricas ---
    tabela, scores = {}, {}
    for exp in ("antes", "depois"):
        hist = pd.read_csv(fontes[exp]["resultados"]).set_index("dataset")
        for k, (arq, nome_csv, nome, _) in CONJUNTOS.items():
            df = ler_scores(fontes[exp]["scores"] / arq)
            scores[(exp, k)] = df
            m = metricas(df)
            tabela[(exp, k)] = m
            h = hist.loc[nome_csv]
            mesma_cm = all(int(h[c]) == m[c] for c in ("tn", "fp", "fn", "tp"))
            check(f"{exp}/{nome}: matriz e AUC recalculadas dos scores = evaluation_results.csv",
                  mesma_cm and abs(h["auc_roc"] - m["auc_roc"]) < 1e-9,
                  f"(AUC csv={h['auc_roc']:.6f}, recalculada={m['auc_roc']:.6f})")
            if "pred_label" in pd.read_csv(fontes[exp]["scores"] / arq, nrows=1).columns:
                diverg = int(((df["prob_pcos"] > THRESHOLD).astype(int) != df["pred_label"]).sum())
                check(f"{exp}/{nome}: pred_label = (prob_pcos > 0,5)", diverg == 0,
                      f"({diverg} divergencias de arredondamento)")

    # --- mesmas imagens, mesma ordem, mesmos rotulos ---
    for k, (_, _, nome, _) in CONJUNTOS.items():
        a, d = scores[("antes", k)], scores[("depois", k)]
        mesmo = len(a) == len(d) and (a["image"].values == d["image"].values).all() \
            and (a["label"].values == d["label"].values).all()
        check(f"{nome}: mesmas imagens, ordem e rotulos antes/depois", mesmo, f"(n antes={len(a)}, depois={len(d)})")
        if not mesmo:
            raise SystemExit("Conjuntos de avaliacao diferentes; comparacao interrompida.")
        pd.DataFrame({
            "indice": np.arange(len(a)), "image": a["image"], "label": a["label"],
            "classe_verdadeira": a["label"].map({0: "normal", 1: "PCOS"}),
            "prob_pcos_antes": a["prob_pcos"], "pred_antes": a["pred_label"],
            "prob_pcos_depois": d["prob_pcos"], "pred_depois": d["pred_label"],
            "logit_antes": a["logit"], "logit_depois": d["logit"],
        }).to_csv(out / f"predicoes_por_imagem_{k}.csv", index=False)

    # --- conferencias de pre-processamento e arquivos ilegiveis ---
    info_depois = json.loads((depois_dir / "avaliacao" / "avaliacao_info.json").read_text(encoding="utf-8"))
    import sys
    sys.path.insert(0, str(RAIZ / "src"))
    from transforms_config import get_eval_transform
    check("depois: transform de avaliacao = get_eval_transform() atual (inalterado pelo PR)",
          info_depois["eval_transform"] == repr(get_eval_transform()))
    check("depois: avaliacao usou o checkpoint do experimento",
          Path(info_depois["checkpoint"]).resolve() == fontes["depois"]["checkpoint"].resolve(),
          f"({info_depois['checkpoint']})")
    check("depois: limiar 0,5", info_depois["threshold"] == THRESHOLD)
    ileg_depois = pd.read_csv(depois_dir / "avaliacao" / "arquivos_ilegiveis.csv")
    check("arquivos ilegiveis (imagem preta) registrados", True,
          f"({len(ileg_depois)} no depois; o historico nao registrou a lista, mas usou o mesmo except)")
    splits_dir = depois_dir / "splits"
    for parte in ("treino", "validacao"):
        a = pd.read_csv(RAIZ / "resultados" / "splits" / f"pcosgen_train_{parte}.csv")
        d = pd.read_csv(splits_dir / f"pcosgen_train_{parte}.csv")
        check(f"split de {parte} identico ao da referencia", a.equals(d), f"(n={len(d)})")

    if args.reavaliacao_antes:
        r_dir = args.reavaliacao_antes.resolve()
        info_r = json.loads((r_dir / "avaliacao_info.json").read_text(encoding="utf-8"))
        check("reavaliacao antes: checkpoint = referencia",
              Path(info_r["checkpoint"]).resolve() == args.antes_checkpoint.resolve(),
              f"(sha256 {info_r['checkpoint_sha256'][:12]}...)")
        ileg_r = pd.read_csv(r_dir / "arquivos_ilegiveis.csv")
        check("arquivos ilegiveis iguais nas duas avaliacoes",
              sorted(ileg_r["image"]) == sorted(ileg_depois["image"]), f"({len(ileg_r)})")
        for k, (arq, _, nome, _) in CONJUNTOS.items():
            h, r = scores[("antes", k)], ler_scores(r_dir / "scores" / arq)
            mesmo = len(h) == len(r) and (h["image"].values == r["image"].values).all()
            dif = float(np.abs(h["logit"].values - r["logit"].values).max()) if mesmo else float("nan")
            npred = int((h["pred_label"].values != r["pred_label"].values).sum()) if mesmo else -1
            check(f"reavaliacao antes/{nome}: reproduz os scores historicos",
                  mesmo and npred == 0 and dif < 1e-3,
                  f"(max |dif. logit|={dif:.2e}, predicoes diferentes={npred}, "
                  f"AUC hist.={tabela[('antes', k)]['auc_roc']:.6f}, reav.={metricas(r)['auc_roc']:.6f})")

    # --- tabelas ---
    linhas = []
    for exp in ("antes", "depois"):
        for k, (_, _, nome, tipo) in CONJUNTOS.items():
            fonte = ("historico (outputs/scores, " if exp == "antes" else "novo (") + "recalculado das predicoes)"
            linhas.append({"experimento": rotulos[exp], "fonte": fonte, "conjunto": nome, "tipo": tipo,
                           **tabela[(exp, k)]})
    pd.DataFrame(linhas).to_csv(out / "metricas_por_conjunto.csv", index=False)

    difs = []
    for k, (_, _, nome, tipo) in CONJUNTOS.items():
        for chave, (rot, pct) in METRICAS.items():
            a, d = tabela[("antes", k)][chave], tabela[("depois", k)][chave]
            difs.append({"conjunto": nome, "tipo": tipo, "metrica": rot, "antes": a, "depois": d,
                         "diferenca_absoluta": d - a,
                         "diferenca_pp": (d - a) * 100 if pct else None,
                         "n": tabela[("depois", k)]["n"]})
    difs = pd.DataFrame(difs)
    difs.to_csv(out / "diferencas_antes_depois.csv", index=False)

    lacunas = []
    for k, (_, _, nome, tipo) in CONJUNTOS.items():
        if tipo != "externo":
            continue
        for chave, (rot, pct) in METRICAS.items():
            row = {"conjunto_externo": nome, "metrica": rot}
            for exp in ("antes", "depois"):
                row[f"interno_{exp}"] = tabela[(exp, "pcosgen_test")][chave]
                row[f"externo_{exp}"] = tabela[(exp, k)][chave]
                row[f"lacuna_{exp}"] = row[f"externo_{exp}"] - row[f"interno_{exp}"]
            row["mudanca_lacuna"] = row["lacuna_depois"] - row["lacuna_antes"]
            row["unidade"] = "proporcao (x100 = p.p.)" if pct else "AUC"
            lacunas.append(row)
    lacunas = pd.DataFrame(lacunas)
    lacunas.to_csv(out / "lacuna_generalizacao.csv", index=False)

    # --- treino ---
    logs, epocas, resumo = {}, {}, []
    for exp in ("antes", "depois"):
        log = pd.read_csv(fontes[exp]["log"])
        logs[exp] = log
        ep, ep_log, fonte_ep = melhor_epoca(fontes[exp]["checkpoint"], log)
        epocas[exp] = ep
        check(f"{exp}: epoca do checkpoint = epoca de maior val_auc no log", ep == ep_log, f"({ep} x {ep_log})")
        b = log[log["epoch"] == ep].iloc[0]
        resumo.append({"experimento": rotulos[exp], "epocas_executadas": len(log), "melhor_epoca": ep,
                       "fonte_melhor_epoca": fonte_ep, "val_auc": b["val_auc"], "val_f1": b["val_f1"],
                       "val_acc": b["val_acc"], "val_loss": b["val_loss"], "train_loss": b["train_loss"],
                       "lr_na_epoca": b["lr"], "log": str(fontes[exp]["log"])})
    pd.DataFrame(resumo).to_csv(out / "resumo_treino.csv", index=False)
    head_only = json.loads((depois_dir / "config.json").read_text(encoding="utf-8")).get("head_only_epochs", 3)

    # --- figuras ---
    fig_curvas_treino(logs, epocas, rotulos, head_only, out / "curvas_treino_antes_depois.png")
    for k, (_, _, nome, _) in CONJUNTOS.items():
        dfs = {e: scores[(e, k)] for e in ("antes", "depois")}
        mets = {e: tabela[(e, k)] for e in ("antes", "depois")}
        fig_roc(dfs, mets, rotulos, nome, out / f"roc_{k}.png")
        fig_matrizes(mets, rotulos, nome, out / f"matriz_confusao_{k}.png")
    fig_generalizacao(tabela, rotulos, out / "generalizacao_interno_externo.png")
    fig_barras_metricas(tabela, rotulos, out / "metricas_antes_depois.png")
    fig_diferencas(tabela, out / "diferencas_antes_depois.png")
    fig_matrizes_todas(tabela, rotulos, out / "matrizes_confusao_todos_conjuntos.png")

    pd.DataFrame(verif).to_csv(out / "verificacoes.csv", index=False)
    escrever_relatorio(out, tabela, difs, lacunas, resumo, verif, depois_dir, args)
    print(f"\nComparacao salva em: {out}")
    if not all(v["ok"] for v in verif):
        print("ATENCAO: ha verificacoes com FALHA; veja verificacoes.csv antes de interpretar.")


def tabela_md(df, casas=4):
    """DataFrame -> tabela Markdown (sem depender do pacote tabulate)."""
    def cel(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.{casas}f}"
        return str(v)
    linhas = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    linhas += ["| " + " | ".join(cel(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(linhas)


def escrever_relatorio(out, tabela, difs, lacunas, resumo, verif, depois_dir, args):
    L = ["# Comparação antes × depois (PR #1)", "",
         f"- Antes: experimento de referência (`{args.antes_checkpoint}`), predições históricas em `{args.antes_scores}`.",
         f"- Depois: `{depois_dir}`.",
         "- Classe positiva: PCOS (1). Limiar fixo: prob_pcos > 0,5 nos dois experimentos.",
         "- Todas as métricas recalculadas das predições por imagem.", "",
         "## Treino", "", tabela_md(pd.DataFrame(resumo).drop(columns=["log"])), "",
         "## Métricas por conjunto", ""]
    for k, (_, _, nome, tipo) in CONJUNTOS.items():
        d = difs[difs["conjunto"] == nome][["metrica", "antes", "depois", "diferenca_absoluta", "diferenca_pp"]]
        L += [f"### {nome} ({tipo}, n = {tabela[('depois', k)]['n']})", "",
              tabela_md(d), ""]
        melhor = d[d["diferenca_absoluta"] > 0]["metrica"].tolist()
        pior = d[d["diferenca_absoluta"] < 0]["metrica"].tolist()
        L += [f"Subiram: {', '.join(melhor) or 'nenhuma'}. Caíram: {', '.join(pior) or 'nenhuma'}.", ""]
    L += ["## Lacuna de generalização (externo − interno)", "",
          "Valor negativo = desempenho externo abaixo do interno. `mudanca_lacuna` > 0 = lacuna menor depois.", "",
          tabela_md(lacunas[["conjunto_externo", "metrica", "lacuna_antes", "lacuna_depois", "mudanca_lacuna"]]), "",
          "## Verificações", "", tabela_md(pd.DataFrame(verif)), "",
          "## Limitações", "",
          "- Uma execução de treino por configuração: não há como estimar variância entre seeds nem "
          "afirmar significância estatística.",
          "- O PR muda várias coisas ao mesmo tempo (duas fases, LRs, augmentation, seeds/cuDNN determinístico); "
          "a diferença não pode ser atribuída a uma mudança isolada.",
          "- Os conjuntos externos têm duplicatas exatas e sobreposição com o PCOSGen-train (ver README); "
          "por orientação, os dados foram usados como estão nos dois experimentos.", ""]
    (out / "RELATORIO.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
