# Comparação antes × depois (PR #1)

- Antes: experimento de referência (`G:\tcc\outputs\models\best_model.pt`), predições históricas em `G:\tcc\outputs\scores`.
- Depois: `G:\tcc\outputs\experimentos\luan_pr1_20261021_142522`.
- Classe positiva: PCOS (1). Limiar fixo: prob_pcos > 0,5 nos dois experimentos.
- Todas as métricas recalculadas das predições por imagem.

## Treino

| experimento | epocas_executadas | melhor_epoca | fonte_melhor_epoca | val_auc | val_f1 | val_acc | val_loss | train_loss | lr_na_epoca |
|---|---|---|---|---|---|---|---|---|---|
| Antes (referência) | 23 | 8 | checkpoint | 0.8347 | 0.7468 | 0.6906 | 0.2644 | 0.2489 | 0.0001 |
| Depois (PR #1) | 24 | 9 | checkpoint | 0.8251 | 0.7244 | 0.6719 | 0.2658 | 0.2634 | 0.0001 |

## Métricas por conjunto

### PCOSGen-test (interno, n = 1468)

| metrica | antes | depois | diferenca_absoluta | diferenca_pp |
|---|---|---|---|---|
| Acurácia | 0.9135 | 0.9067 | -0.0068 | -0.6812 |
| Precisão | 0.9278 | 0.9328 | 0.0050 | 0.4987 |
| Sensibilidade (recall PCOS) | 0.9534 | 0.9372 | -0.0162 | -1.6175 |
| Especificidade | 0.8129 | 0.8297 | 0.0168 | 1.6787 |
| F1-score | 0.9404 | 0.9350 | -0.0054 | -0.5425 |
| AUC-ROC | 0.9599 | 0.9572 | -0.0027 |  |

Subiram: Precisão, Especificidade. Caíram: Acurácia, Sensibilidade (recall PCOS), F1-score, AUC-ROC.

### Kaggle (externo, n = 3856)

| metrica | antes | depois | diferenca_absoluta | diferenca_pp |
|---|---|---|---|---|
| Acurácia | 0.5272 | 0.5615 | 0.0342 | 3.4232 |
| Precisão | 0.4512 | 0.4753 | 0.0241 | 2.4143 |
| Sensibilidade (recall PCOS) | 0.7513 | 0.7551 | 0.0038 | 0.3827 |
| Especificidade | 0.3737 | 0.4288 | 0.0551 | 5.5070 |
| F1-score | 0.5638 | 0.5834 | 0.0196 | 1.9624 |
| AUC-ROC | 0.7164 | 0.7263 | 0.0099 |  |

Subiram: Acurácia, Precisão, Sensibilidade (recall PCOS), Especificidade, F1-score, AUC-ROC. Caíram: nenhuma.

### Figshare (externo, n = 11784)

| metrica | antes | depois | diferenca_absoluta | diferenca_pp |
|---|---|---|---|---|
| Acurácia | 0.6237 | 0.6166 | -0.0071 | -0.7128 |
| Precisão | 0.6573 | 0.6536 | -0.0038 | -0.3773 |
| Sensibilidade (recall PCOS) | 0.7236 | 0.7108 | -0.0128 | -1.2824 |
| Especificidade | 0.4882 | 0.4888 | 0.0006 | 0.0600 |
| F1-score | 0.6889 | 0.6810 | -0.0079 | -0.7909 |
| AUC-ROC | 0.6802 | 0.6781 | -0.0021 |  |

Subiram: Especificidade. Caíram: Acurácia, Precisão, Sensibilidade (recall PCOS), F1-score, AUC-ROC.

## Lacuna de generalização (externo − interno)

Valor negativo = desempenho externo abaixo do interno. `mudanca_lacuna` > 0 = lacuna menor depois.

| conjunto_externo | metrica | lacuna_antes | lacuna_depois | mudanca_lacuna |
|---|---|---|---|---|
| Kaggle | Acurácia | -0.3863 | -0.3452 | 0.0410 |
| Kaggle | Precisão | -0.4766 | -0.4575 | 0.0192 |
| Kaggle | Sensibilidade (recall PCOS) | -0.2021 | -0.1821 | 0.0200 |
| Kaggle | Especificidade | -0.4393 | -0.4010 | 0.0383 |
| Kaggle | F1-score | -0.3766 | -0.3516 | 0.0250 |
| Kaggle | AUC-ROC | -0.2435 | -0.2309 | 0.0126 |
| Figshare | Acurácia | -0.2898 | -0.2901 | -0.0003 |
| Figshare | Precisão | -0.2704 | -0.2792 | -0.0088 |
| Figshare | Sensibilidade (recall PCOS) | -0.2298 | -0.2264 | 0.0034 |
| Figshare | Especificidade | -0.3247 | -0.3409 | -0.0162 |
| Figshare | F1-score | -0.2515 | -0.2540 | -0.0025 |
| Figshare | AUC-ROC | -0.2797 | -0.2791 | 0.0006 |

## Verificações

| verificacao | ok | detalhe |
|---|---|---|
| antes/PCOSGen-test: matriz e AUC recalculadas dos scores = evaluation_results.csv | True | (AUC csv=0.959925, recalculada=0.959925) |
| antes/Kaggle: matriz e AUC recalculadas dos scores = evaluation_results.csv | True | (AUC csv=0.716382, recalculada=0.716382) |
| antes/Figshare: matriz e AUC recalculadas dos scores = evaluation_results.csv | True | (AUC csv=0.680230, recalculada=0.680230) |
| depois/PCOSGen-test: matriz e AUC recalculadas dos scores = evaluation_results.csv | True | (AUC csv=0.957205, recalculada=0.957205) |
| depois/PCOSGen-test: pred_label = (prob_pcos > 0,5) | True | (0 divergencias de arredondamento) |
| depois/Kaggle: matriz e AUC recalculadas dos scores = evaluation_results.csv | True | (AUC csv=0.726264, recalculada=0.726264) |
| depois/Kaggle: pred_label = (prob_pcos > 0,5) | True | (0 divergencias de arredondamento) |
| depois/Figshare: matriz e AUC recalculadas dos scores = evaluation_results.csv | True | (AUC csv=0.678095, recalculada=0.678095) |
| depois/Figshare: pred_label = (prob_pcos > 0,5) | True | (0 divergencias de arredondamento) |
| PCOSGen-test: mesmas imagens, ordem e rotulos antes/depois | True | (n antes=1468, depois=1468) |
| Kaggle: mesmas imagens, ordem e rotulos antes/depois | True | (n antes=3856, depois=3856) |
| Figshare: mesmas imagens, ordem e rotulos antes/depois | True | (n antes=11784, depois=11784) |
| depois: transform de avaliacao = get_eval_transform() atual (inalterado pelo PR) | True |  |
| depois: avaliacao usou o checkpoint do experimento | True | (G:\tcc\outputs\experimentos\luan_pr1_20261021_142522\models\best_model.pt) |
| depois: limiar 0,5 | True |  |
| arquivos ilegiveis (imagem preta) registrados | True | (10 no depois; o historico nao registrou a lista, mas usou o mesmo except) |
| split de treino identico ao da referencia | True | (n=2560) |
| split de validacao identico ao da referencia | True | (n=640) |
| reavaliacao antes: checkpoint = referencia | True | (sha256 d9839653a436...) |
| arquivos ilegiveis iguais nas duas avaliacoes | True | (10) |
| reavaliacao antes/PCOSGen-test: reproduz os scores historicos | True | (max |dif. logit|=0.00e+00, predicoes diferentes=0, AUC hist.=0.959925, reav.=0.959925) |
| reavaliacao antes/Kaggle: reproduz os scores historicos | True | (max |dif. logit|=0.00e+00, predicoes diferentes=0, AUC hist.=0.716382, reav.=0.716382) |
| reavaliacao antes/Figshare: reproduz os scores historicos | True | (max |dif. logit|=0.00e+00, predicoes diferentes=0, AUC hist.=0.680230, reav.=0.680230) |
| antes: epoca do checkpoint = epoca de maior val_auc no log | True | (8 x 8) |
| depois: epoca do checkpoint = epoca de maior val_auc no log | True | (9 x 9) |

## Limitações

- Uma execução de treino por configuração: não há como estimar variância entre seeds nem afirmar significância estatística.
- O PR muda várias coisas ao mesmo tempo (duas fases, LRs, augmentation, seeds/cuDNN determinístico); a diferença não pode ser atribuída a uma mudança isolada.
- Os conjuntos externos têm duplicatas exatas e sobreposição com o PCOSGen-train (ver README); por orientação, os dados foram usados como estão nos dois experimentos.
