"""
dedup.py
Detecta imagens duplicadas/quase duplicadas via perceptual hash (pHash).

- Dentro do PCOSGen-train: agrupa imagens a <= THRESHOLD bits de distancia
  (union-find). Os grupos sao usados na divisao treino/validacao por grupo
  (dataset.split_train_val_grupos), para que duplicatas fiquem do mesmo lado.
- PCOSGen-test, Kaggle e Figshare: menor distancia de cada imagem ate qualquer
  imagem do PCOSGen-train. Imagens a <= THRESHOLD sao marcadas como duplicatas
  do treino; evaluate.py reporta as metricas com e sem elas.

Para cobrir copias espelhadas ou rotacionadas, o hash de cada imagem e'
calculado tambem para flip horizontal e rotacoes de 90/180/270 graus; a
distancia entre duas imagens e' a menor entre essas versoes e o hash original
da outra.

Nenhum arquivo de imagem ou rotulo e' alterado.

Saidas em outputs/dedup/:
- pcosgen_train_groups.csv: imagePath, label, group_id, group_size, grupo_rotulos_mistos
- cross_overlap.csv: conjunto, image, label, min_dist, train_mais_proximo,
  label_train_mais_proximo, duplicata, rotulo_discorda
- resumo.csv e params.json

Dependencia: pip install imagehash
"""

import json
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from dataset import load_pcosgen_labels

THRESHOLD = 6  # distancia de Hamming maxima (0-64) para considerar duplicata
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PRE_SIZE = 64  # reduz antes das variantes; o pHash reduz para 32x32 de qualquer forma

TRAIN_LABELS = Path("G:/tcc/data/pcosgen/train/PCOSGen-train/PCOSGen-train/class_label.xlsx")
TRAIN_IMG_DIR = Path("G:/tcc/data/pcosgen/train/PCOSGen-train/PCOSGen-train/images")
TEST_LABELS = Path("G:/tcc/data/pcosgen/test/class label.csv")
TEST_IMG_DIR = Path("G:/tcc/data/pcosgen/test/PCOSGen-test/images")
KAGGLE_DIR = Path("G:/tcc/data/kaggle/data")
FIGSHARE_DIR = Path("G:/tcc/data/figshare/PCOS")
OUT_DIR = Path("G:/tcc/outputs/dedup")


def phash_uint64(img):
    """pHash de 64 bits como inteiro sem sinal."""
    bits = imagehash.phash(img).hash.flatten().astype(np.uint8)
    return np.packbits(bits).view(">u8")[0].astype(np.uint64)


def hash_variants(path):
    """Hashes da imagem original, espelhada e rotacionada (90/180/270)."""
    img = Image.open(path).convert("L").resize((PRE_SIZE, PRE_SIZE), Image.LANCZOS)
    variants = [img, img.transpose(Image.FLIP_LEFT_RIGHT),
                img.rotate(90), img.rotate(180), img.rotate(270)]
    return np.array([phash_uint64(v) for v in variants], dtype=np.uint64)


def hash_all(paths, nome):
    hashes, ok = [], []
    for i, p in enumerate(paths):
        try:
            hashes.append(hash_variants(p))
            ok.append(True)
        except Exception:
            hashes.append(np.zeros(5, dtype=np.uint64))
            ok.append(False)
        if (i + 1) % 2000 == 0:
            print(f"  {nome}: {i + 1}/{len(paths)}")
    return np.stack(hashes), np.array(ok)


def min_dist_to(base_hashes, query_variants):
    """Para cada query, menor distancia (entre suas variantes) ate cada base.
    Retorna matriz (n_query, n_base)."""
    d = None
    for v in range(query_variants.shape[1]):
        x = np.bitwise_count(np.bitwise_xor(query_variants[:, v][:, None], base_hashes[None, :])).astype(np.int16)
        d = x if d is None else np.minimum(d, x)
    return d


def group_duplicates(hashes, ok, threshold=THRESHOLD):
    """Union-find: imagens a <= threshold bits de distancia ficam no mesmo grupo."""
    n = len(hashes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, n, 500):
        d = min_dist_to(hashes[:, 0], hashes[start:start + 500])
        for a, b in zip(*np.nonzero(d <= threshold)):
            i, j = start + a, b
            if i != j and ok[i] and ok[j]:
                parent[find(j)] = find(i)
    return np.array([find(i) for i in range(n)])


def folder_samples(root_dirs, class_map, prefix_parts=3):
    rows = []
    for class_name, folders in root_dirs.items():
        for folder in folders:
            for p in sorted(Path(folder).iterdir()):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    rows.append({"image": "/".join(p.parts[-prefix_parts:]), "path": p,
                                 "label": class_map[class_name]})
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- PCOSGen-train: grupos de duplicatas ---
    train = load_pcosgen_labels(TRAIN_LABELS)
    train_paths = [TRAIN_IMG_DIR / str(x) for x in train["imagePath"]]
    print(f"PCOSGen-train: {len(train)} imagens")
    h_train, ok_train = hash_all(train_paths, "PCOSGen-train")
    np.savez(OUT_DIR / "hashes_pcosgen_train.npz", hashes=h_train, ok=ok_train,
             imagePath=train["imagePath"].astype(str).values)
    groups = group_duplicates(h_train, ok_train)
    train["group_id"] = groups
    train["group_size"] = train.groupby("group_id")["imagePath"].transform("size")
    train["grupo_rotulos_mistos"] = train.groupby("group_id")["label"].transform("nunique") > 1
    train["hash_ok"] = ok_train
    train.to_csv(OUT_DIR / "pcosgen_train_groups.csv", index=False)

    # --- conjuntos de avaliacao x treino ---
    test = load_pcosgen_labels(TEST_LABELS)
    existentes = {p.name for p in TEST_IMG_DIR.iterdir() if p.is_file()}
    test = test[test["imagePath"].isin(existentes)].reset_index(drop=True)
    test = pd.DataFrame({"image": test["imagePath"].astype(str),
                         "path": [TEST_IMG_DIR / str(x) for x in test["imagePath"]],
                         "label": test["label"]})
    kaggle = folder_samples(
        {"notinfected": [KAGGLE_DIR / "train" / "notinfected", KAGGLE_DIR / "test" / "notinfected"],
         "infected": [KAGGLE_DIR / "train" / "infected", KAGGLE_DIR / "test" / "infected"]},
        {"notinfected": 0, "infected": 1})
    figshare = folder_samples(
        {"noninfected": [FIGSHARE_DIR / "noninfected"], "infected": [FIGSHARE_DIR / "infected"]},
        {"noninfected": 0, "infected": 1})

    base = h_train[:, 0]
    rows, resumo = [], []
    for nome, df in [("PCOSGen-test", test), ("Kaggle", kaggle), ("Figshare", figshare)]:
        print(f"{nome}: {len(df)} imagens")
        h, ok = hash_all(list(df["path"]), nome)
        np.savez(OUT_DIR / f"hashes_{nome.lower().replace('-', '_')}.npz", hashes=h, ok=ok,
                 image=df["image"].astype(str).values)
        mins, idx = np.empty(len(df), dtype=int), np.empty(len(df), dtype=int)
        for start in range(0, len(df), 500):
            d = min_dist_to(base, h[start:start + 500])
            d[:, ~ok_train] = 999
            idx[start:start + 500] = d.argmin(axis=1)
            mins[start:start + 500] = d.min(axis=1)
        mins[~ok] = -1  # ilegivel
        out = pd.DataFrame({
            "conjunto": nome, "image": df["image"], "label": df["label"].astype(int),
            "min_dist": mins,
            "train_mais_proximo": train["imagePath"].values[idx],
            "label_train_mais_proximo": train["label"].values[idx],
        })
        out["duplicata"] = (out["min_dist"] >= 0) & (out["min_dist"] <= THRESHOLD)
        out["rotulo_discorda"] = out["duplicata"] & (out["label"] != out["label_train_mais_proximo"])
        out.loc[~ok, ["train_mais_proximo", "label_train_mais_proximo"]] = [None, None]
        rows.append(out)
        resumo.append({"conjunto": nome, "n": len(out), "ilegiveis": int((~ok).sum()),
                       "duplicatas_do_treino": int(out["duplicata"].sum()),
                       "pct_duplicatas": 100 * out["duplicata"].mean(),
                       "duplicatas_rotulo_discorda": int(out["rotulo_discorda"].sum())})
    pd.concat(rows).to_csv(OUT_DIR / "cross_overlap.csv", index=False)

    n_grupos_dup = int((train.drop_duplicates("group_id")["group_size"] > 1).sum())
    resumo.insert(0, {"conjunto": "PCOSGen-train (interno)", "n": len(train),
                      "ilegiveis": int((~ok_train).sum()),
                      "grupos_com_duplicatas": n_grupos_dup,
                      "imagens_em_grupos_com_duplicatas": int((train["group_size"] > 1).sum()),
                      "grupos_rotulos_mistos": int(train.loc[train["grupo_rotulos_mistos"], "group_id"].nunique())})
    resumo = pd.DataFrame(resumo)
    resumo.to_csv(OUT_DIR / "resumo.csv", index=False)
    with open(OUT_DIR / "params.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": THRESHOLD, "hash": "imagehash.phash 64 bits, escala de cinza",
                   "pre_resize": PRE_SIZE, "variantes": ["original", "flip_h", "rot90", "rot180", "rot270"],
                   "imagehash": imagehash.__version__}, f, indent=2, ensure_ascii=False)
    print(resumo.to_string(index=False))
    print(f"\nSaidas em {OUT_DIR}")


if __name__ == "__main__":
    main()
