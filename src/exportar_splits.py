"""
exportar_splits.py
Salva em arquivo a particao treino/validacao do PCOSGen-train usada no
treinamento, gerada por split_train_val (80/20 estratificado, seed 42), igual
ao que train.py faz.

Os conjuntos de teste (PCOSGen-test, Kaggle, Figshare) sao usados inteiros; a
lista de imagens de cada um esta nos CSVs de scores salvos por evaluate.py.

Salva em resultados/splits/:
- pcosgen_train_treino.csv
- pcosgen_train_validacao.csv
Colunas: imagePath, label (0 = normal, 1 = PCOS).
"""

from pathlib import Path

from dataset import load_pcosgen_train_labels, split_train_val

LABELS_PATH = Path("G:/tcc/data/pcosgen/train/PCOSGen-train/PCOSGen-train/class_label.xlsx")
SAIDA = Path("G:/tcc/resultados/splits")
SEED = 42  # mesmo valor de train.py


def main():
    df = load_pcosgen_train_labels(LABELS_PATH)
    train_df, val_df = split_train_val(df, val_size=0.2, seed=SEED)

    SAIDA.mkdir(parents=True, exist_ok=True)
    for nome, parte in [("treino", train_df), ("validacao", val_df)]:
        caminho = SAIDA / f"pcosgen_train_{nome}.csv"
        parte.to_csv(caminho, index=False)
        contagem = parte["label"].value_counts().to_dict()
        print(f"{nome}: {len(parte)} imagens "
              f"(normal={contagem.get(0, 0)}, PCOS={contagem.get(1, 0)}) -> {caminho}")


if __name__ == "__main__":
    main()
