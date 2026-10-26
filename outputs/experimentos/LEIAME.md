# Experimentos: referência × PR #1 (Luan)

- `referencia_20260620/manifesto_referencia.json`: lista e SHA-256 dos arquivos do
  experimento de referência, que **continuam** em `outputs/models/`, `outputs/logs/`,
  `outputs/scores/` e `resultados/`. Nada foi copiado nem movido.
- `luan_pr1_AAAAMMDD_HHMMSS/`: experimento novo (criado por `src/train.py`).
- `referencia_reavaliacao_AAAAMMDD_HHMMSS/`: checkpoint antigo avaliado de novo
  (opcional, só para conferir que os scores históricos se reproduzem).

Comandos (PowerShell, na pasta `G:\tcc`):

```powershell
# 1. treino (~50 s por época na GPU; até 50 épocas)
$EXP = "G:\tcc\outputs\experimentos\luan_pr1_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory -Force $EXP | Out-Null
.\venv\Scripts\python.exe src\train.py --out-dir $EXP | Tee-Object "$EXP\treino_console.txt"

# 2. avaliação do checkpoint novo
.\venv\Scripts\python.exe src\evaluate.py --checkpoint "$EXP\models\best_model.pt" --out-dir "$EXP\avaliacao" | Tee-Object "$EXP\avaliacao_console.txt"

# 3. (opcional) reavaliação do checkpoint antigo, em pasta separada
$REAV = "G:\tcc\outputs\experimentos\referencia_reavaliacao_$(Get-Date -Format yyyyMMdd_HHmmss)"
.\venv\Scripts\python.exe src\evaluate.py --checkpoint outputs\models\best_model.pt --out-dir $REAV

# 4. comparação (tabelas, gráficos, RELATORIO.md em $EXP\comparacao)
.\venv\Scripts\python.exe src\comparar_experimentos.py --depois $EXP --reavaliacao-antes $REAV
# sem o passo 3: omita --reavaliacao-antes
```
