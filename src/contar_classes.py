from pathlib import Path
import pandas as pd

BASE = Path(r"G:\tcc\data")

FIGSHARE_DIR     = BASE / "figshare" / "PCOS"
KAGGLE_DIR       = BASE / "kaggle" / "data"
PCOSGEN_TRAIN_XL = BASE / "pcosgen" / "train" / "PCOSGen-train" / "PCOSGen-train" / "class_label.xlsx"
PCOSGEN_TEST_CSV = BASE / "pcosgen" / "test" / "class label.csv"

# Nome da coluna de rotulo nos arquivos do PCOSGen. Se o script reclamar,
# rode uma vez e veja a lista de colunas que ele imprime, e ajuste aqui.
PCOSGEN_LABEL_COL = "Healthy"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def contar_imagens_em_pasta(pasta: Path) -> int:
    """Conta arquivos de imagem dentro de uma pasta (nao recursivo)."""
    if not pasta.exists():
        print(f"  [aviso] pasta nao encontrada: {pasta}")
        return 0
    return sum(1 for p in pasta.iterdir() if p.suffix.lower() in IMG_EXTS)


def eh_pasta_normal(nome: str) -> bool:
    n = nome.lower()
    return ("noninf" in n) or ("notinf" in n) or n.startswith("non") or n.startswith("not")


def contar_dataset_por_pastas(raiz: Path, recursivo_em_subpastas=False):
    """
    Conta normal x PCOS para datasets organizados em pastas por classe.
    Se recursivo_em_subpastas=True, soma todas as subpastas de classe encontradas
    em qualquer nivel (util para Kaggle, que tem train/ e test/).
    """
    normal = pcos = 0
    if not raiz.exists():
        print(f"  [aviso] raiz nao encontrada: {raiz}")
        return normal, pcos

    if recursivo_em_subpastas:
        # procura todas as pastas cujo nome indique classe
        for pasta in raiz.rglob("*"):
            if pasta.is_dir() and ("inf" in pasta.name.lower()):
                if eh_pasta_normal(pasta.name):
                    normal += contar_imagens_em_pasta(pasta)
                else:
                    pcos += contar_imagens_em_pasta(pasta)
    else:
        for pasta in raiz.iterdir():
            if pasta.is_dir() and ("inf" in pasta.name.lower()):
                if eh_pasta_normal(pasta.name):
                    normal += contar_imagens_em_pasta(pasta)
                else:
                    pcos += contar_imagens_em_pasta(pasta)
    return normal, pcos


def contar_pcosgen(arquivo: Path):
    """Le o arquivo de rotulos do PCOSGen e conta normal (Healthy=1) x PCOS (Healthy=0)."""
    if not arquivo.exists():
        print(f"  [aviso] arquivo de rotulos nao encontrado: {arquivo}")
        return 0, 0

    if arquivo.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_csv(arquivo) if arquivo.suffix.lower() == ".csv" else pd.read_excel(arquivo)
    else:
        df = pd.read_csv(arquivo)

    if PCOSGEN_LABEL_COL not in df.columns:
        print(f"  [aviso] coluna '{PCOSGEN_LABEL_COL}' nao encontrada em {arquivo.name}.")
        print(f"          colunas disponiveis: {list(df.columns)}")
        return 0, 0

    col = df[PCOSGEN_LABEL_COL]
    normal = int((col == 1).sum())  # Healthy = 1 -> normal
    pcos   = int((col == 0).sum())  # Healthy = 0 -> PCOS
    return normal, pcos


def linha(nome, normal, pcos):
    total = normal + pcos
    if total == 0:
        prop = "  -"
    else:
        prop = f"{normal/total*100:4.1f}% / {pcos/total*100:4.1f}%"
    print(f"{nome:<16} {normal:>7} {pcos:>7} {total:>7}   {prop}")


def main():
    print("\nContando imagens por classe...\n")

    print("Figshare:")
    fig_n, fig_p = contar_dataset_por_pastas(FIGSHARE_DIR, recursivo_em_subpastas=True)

    print("Kaggle:")
    kag_n, kag_p = contar_dataset_por_pastas(KAGGLE_DIR, recursivo_em_subpastas=True)

    print("PCOSGen-train:")
    ptr_n, ptr_p = contar_pcosgen(PCOSGEN_TRAIN_XL)

    print("PCOSGen-test:")
    pte_n, pte_p = contar_pcosgen(PCOSGEN_TEST_CSV)

    print("\n" + "=" * 52)
    print(f"{'Conjunto':<16} {'Normal':>7} {'PCOS':>7} {'Total':>7}   Normal/PCOS")
    print("-" * 52)
    linha("PCOSGen-train", ptr_n, ptr_p)
    linha("PCOSGen-test", pte_n, pte_p)
    linha("Kaggle", kag_n, kag_p)
    linha("Figshare", fig_n, fig_p)
    print("=" * 52)
    print("\nUse as linhas de PCOSGen-test e Figshare para preencher a Tabela 1.\n")


if __name__ == "__main__":
    main()