"""
verificar_duplicatas.py
Procura imagens duplicadas (arquivos identicos, via MD5) dentro de cada base e
entre o PCOSGen-train e as bases de teste, e recalcula as metricas de cada
conjunto de teste (a) mantendo uma unica copia de cada imagem e (b) removendo
tambem as imagens identicas a alguma do PCOSGen-train. Grupos de arquivos
identicos com rotulos diferentes sao descartados nas versoes (a) e (b).

Usa os scores por imagem salvos por evaluate.py (outputs/scores/), entao nao
precisa do modelo. Detecta apenas copias exatas, nao imagens parecidas.

Salva:
- outputs/logs/duplicatas_grupos.csv   : cada grupo de arquivos identicos
- outputs/logs/metricas_sem_duplicatas.csv
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix

DATA = Path("G:/tcc/data")
SCORES_DIR = Path("G:/tcc/outputs/scores")
LOGS_DIR = Path("G:/tcc/outputs/logs")

PCOSGEN_TRAIN_IMG = DATA / "pcosgen/train/PCOSGen-train/PCOSGen-train/images"
PCOSGEN_TEST_IMG = DATA / "pcosgen/test/PCOSGen-test/images"
KAGGLE_DIR = DATA / "kaggle/data"
FIGSHARE_DIR = DATA / "figshare/PCOS"

# conjunto de teste -> (CSV de scores, funcao que leva o id salvo ao arquivo)
TESTES = {
    "PCOSGen-test": ("scores_pcosgen_test.csv", lambda i: PCOSGEN_TEST_IMG / i),
    "Kaggle": ("scores_kaggle.csv", lambda i: KAGGLE_DIR / i),
    "Figshare": ("scores_figshare.csv", lambda i: FIGSHARE_DIR.parent / i),
}


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def metricas(labels, logits):
    preds = (logits > 0).astype(int)  # equivale a sigmoid(logit) > 0.5
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "auc_roc": roc_auc_score(labels, logits),
        "f1_score": f1_score(labels, preds),
        "accuracy": accuracy_score(labels, preds),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    print("Calculando hashes do PCOSGen-train...")
    hashes_treino = {md5(p) for p in PCOSGEN_TRAIN_IMG.iterdir() if p.is_file()}

    grupos, linhas = [], []
    for nome, (arquivo, caminho) in TESTES.items():
        print(f"Calculando hashes de {nome}...")
        s = pd.read_csv(SCORES_DIR / arquivo)
        s["md5"] = [md5(caminho(i)) for i in s["image"]]

        # grupos de arquivos identicos dentro do conjunto
        for h, g in s.groupby("md5"):
            if g["image"].nunique() > 1:
                grupos.append({
                    "conjunto": nome,
                    "md5": h,
                    "n_arquivos": g["image"].nunique(),
                    "rotulos": "/".join(map(str, sorted(g["label"].unique()))),
                    "arquivos": ";".join(sorted(g["image"].unique())),
                })

        # uma linha por imagem unica; rotulos conflitantes sao descartados
        por_hash = s.groupby("md5").agg(
            label=("label", "first"), n_rotulos=("label", "nunique"),
            logit=("logit", "first"),
        )
        conflitos = int((por_hash["n_rotulos"] > 1).sum())
        unicos = por_hash[por_hash["n_rotulos"] == 1]
        fora_do_treino = unicos[~unicos.index.isin(hashes_treino)]

        versoes = [
            ("original", s),
            ("sem duplicatas", unicos),
            ("sem duplicatas e sem imagens do treino", fora_do_treino),
        ]
        for versao, dados in versoes:
            labels = dados["label"].to_numpy()
            hashes = dados["md5"] if "md5" in dados else dados.index.to_series()
            linhas.append({
                "conjunto": nome,
                "versao": versao,
                "n_samples": len(dados),
                "n_normal": int((labels == 0).sum()),
                "n_pcos": int((labels == 1).sum()),
                "grupos_rotulo_conflitante_descartados":
                    conflitos if versao != "original" else 0,
                "iguais_a_imagem_do_treino": int(hashes.isin(hashes_treino).sum()),
                **metricas(labels, dados["logit"].to_numpy()),
            })

    pd.DataFrame(grupos).to_csv(LOGS_DIR / "duplicatas_grupos.csv", index=False)
    resultado = pd.DataFrame(linhas)
    resultado.to_csv(LOGS_DIR / "metricas_sem_duplicatas.csv", index=False)

    pd.set_option("display.width", 200)
    print()
    print(resultado.round(4).to_string(index=False))
    print(f"\nSalvo em: {LOGS_DIR / 'metricas_sem_duplicatas.csv'}")


if __name__ == "__main__":
    main()
