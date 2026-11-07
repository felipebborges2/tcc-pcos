"""
rodar_ablacao.py
Roda a ablacao do relatorio tecnico (secao 6), um passo por vez. Cada passo
mantem as mudancas dos anteriores, tem pasta propria e gera so' as figuras
dele (gerar_figuras_experimento.py), sem comparacao com outros resultados.

  0  modelo do PR #1 reavaliado com arquivos ilegiveis descartados   (sem treino)
  1  + limiar de Youden calibrado na validacao                        (sem treino)
  2  + divisao treino/validacao por grupo de duplicatas (dedup.py)
  3  + escala de cinza + PadToSquare
  4  + augmentation para ultrassom
  5  + AdamW, LR por camada e descongelamento gradual (selecao por val_loss)
  6  + label smoothing 0,05

Pre-requisito: outputs/dedup/ gerado por dedup.py.

Uso:
    python src/rodar_ablacao.py                       # cria outputs/experimentos/ablacao_<data_hora>
    python src/rodar_ablacao.py --root <pasta>        # retoma uma ablacao interrompida
    python src/rodar_ablacao.py --passos 0 1          # so' alguns passos
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SRC = Path(__file__).resolve().parent
RAIZ = SRC.parent
PY = sys.executable
BASE_PR1 = RAIZ / "outputs" / "experimentos" / "luan_pr1_20261021_142522"

PASSOS = [
    ("passo0_avaliacao_corrigida", "Passo 0 — PR #1 reavaliado, ilegíveis descartados", None),
    ("passo1_limiar_youden", "Passo 1 — + limiar de Youden (validação)", None),
    ("passo2_split_grupo", "Passo 2 — + divisão por grupo de duplicatas",
     ["--split", "grupo"]),
    ("passo3_gray_pad", "Passo 3 — + escala de cinza e PadToSquare",
     ["--split", "grupo", "--preproc", "gray_pad"]),
    ("passo4_aug_ultrassom", "Passo 4 — + augmentation para ultrassom",
     ["--split", "grupo", "--preproc", "gray_pad", "--augment", "ultrassom"]),
    ("passo5_adamw_gradual", "Passo 5 — + AdamW, LR por camada, descongelamento gradual",
     ["--split", "grupo", "--preproc", "gray_pad", "--augment", "ultrassom",
      "--otimizacao", "adamw_gradual"]),
    ("passo6_label_smoothing", "Passo 6 — + label smoothing 0,05",
     ["--split", "grupo", "--preproc", "gray_pad", "--augment", "ultrassom",
      "--otimizacao", "adamw_gradual", "--label-smoothing", "0.05"]),
]


def run(cmd, log_file):
    """Roda um comando mostrando e gravando a saida; aborta se falhar."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    with open(log_file, "a", encoding="utf-8") as log:
        proc = subprocess.Popen([str(c) for c in cmd], cwd=SRC, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"Comando falhou (codigo {proc.returncode}); veja {log_file}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--passos", type=int, nargs="*", default=list(range(len(PASSOS))))
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-amostras", type=int, default=0, help="so' para teste rapido")
    p.add_argument("--epocas", type=int, default=None, help="so' para teste rapido")
    args = p.parse_args()

    if not (RAIZ / "outputs" / "dedup" / "cross_overlap.csv").exists():
        raise SystemExit("Rode antes: python src/dedup.py")
    root = (args.root or RAIZ / "outputs" / "experimentos"
            / f"ablacao_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"Pasta da ablacao: {root}")
    debug = ["--debug-amostras", str(args.debug_amostras)] if args.debug_amostras else []
    workers = ["--num-workers", str(args.num_workers)]

    for i in args.passos:
        nome, titulo, train_args = PASSOS[i]
        exp = root / nome
        if (exp / "figuras" / "matrizes_confusao.png").exists():
            print(f"\n[{nome}] ja concluido, pulando")
            continue
        exp.mkdir(exist_ok=True)
        console = exp / "console.txt"
        print(f"\n{'#' * 70}\n# {titulo}\n{'#' * 70}", flush=True)

        log_arg = []
        limiar = ["--limiar", "checkpoint"]
        if train_args is None:
            ckpt = BASE_PR1 / "models" / "best_model.pt"
            log_arg = ["--log", BASE_PR1 / "logs" / "train_log.csv"]
            if i == 1:
                limiar_json = exp / "limiar.json"
                if not limiar_json.exists():
                    run([PY, "-u", "calibrar_limiar.py", "--checkpoint", ckpt, "--out", limiar_json]
                        + workers, console)
                limiar = ["--limiar", limiar_json]
            else:
                limiar = ["--limiar", "0.5"]
        else:
            ckpt = exp / "models" / "best_model.pt"
            cfg = exp / "config.json"
            terminado = cfg.exists() and "fim" in json.loads(cfg.read_text(encoding="utf-8"))
            if not terminado:
                if ckpt.exists() or (exp / "logs" / "train_log.csv").exists():
                    raise SystemExit(f"{exp} tem um treino incompleto; apague ou renomeie essa pasta "
                                     "para treinar de novo.")
                epocas = ["--epocas", str(args.epocas)] if args.epocas else []
                run([PY, "-u", "train.py", "--out-dir", exp] + train_args + workers + debug + epocas,
                    console)

        if not (exp / "avaliacao" / "evaluation_results.csv").exists():
            run([PY, "-u", "evaluate.py", "--checkpoint", ckpt, "--out-dir", exp / "avaliacao"]
                + limiar + workers + debug, console)
        run([PY, "-u", "gerar_figuras_experimento.py", "--exp", exp, "--titulo", titulo] + log_arg,
            console)

    print(f"\nAblacao concluida: {root}")


if __name__ == "__main__":
    main()
