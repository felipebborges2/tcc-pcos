"""Diagnosticos de treino/validacao interna, sem consultar avaliacao externa.

``out_dir`` de save_augmentation_examples e a raiz do experimento.
Comparacao usa splits/pcosgen_train_{treino,validacao}.csv, config.json
(melhor_epoca, fim, epocas_executadas) e logs/train_log.csv como fonte canonica.
Metricas de classificacao nos logs usam prob_pcos > 0.5; o limiar de Youden
e reportado separadamente, nunca aplicado retroativamente a essas metricas.

Opcionalmente, diagnostics/val_predictions.csv ou best_validation_scores.csv
pode preencher metricas ausentes.
Exige imagePath, label, epoch e logit (ou prob_pcos), cobrindo exatamente a
validacao interna na melhor epoca. Nao procura scores em outras pastas.
Logs antigos sem predicoes permitem recuperacao apenas quando acc/F1 exatos
e contagens do split determinam uma unica matriz de confusao inteira.

CLI: python src/diagnosticos_treino.py --passo6 PATH --passo7 PATH
"""

import argparse
import json
import random
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


_METRICS = {
    "best_epoch": (),
    "best_val_loss": ("val_loss",),
    "val_auc": ("val_auc",),
    "val_accuracy": ("val_acc", "val_accuracy"),
    "val_balanced_accuracy": ("val_balanced_acc", "val_balanced_accuracy"),
    "val_precision": ("val_precision",),
    "val_recall": ("val_recall",),
    "val_f1": ("val_f1",),
    "threshold_youden": (),
    "val_specificity": ("val_specificity",),
    "train_val_auc_gap": (),
}


def _binary_labels(labels):
    values = (labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor)
              else np.asarray(labels))
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Rotulos devem ser um vetor finito de 0 e 1.")
    if set(np.unique(values)) != {0, 1}:
        raise ValueError("Sao obrigatorias ambas as classes: 0=normal, 1=PCOS.")
    return values


def _probabilities(logits, n):
    values = (logits.detach().to(device="cpu", dtype=torch.float32)
              if isinstance(logits, torch.Tensor)
              else torch.from_numpy(np.array(logits, dtype=np.float32, copy=True)))
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or len(values) != n or not torch.isfinite(values).all():
        raise ValueError("Logits devem ser finitos, um por rotulo (vetor ou Nx1).")
    return torch.sigmoid(values).numpy()


def probability_diagnostics(labels, logits):
    """Media/desvio populacional de P(PCOS), separados pela classe verdadeira."""
    labels = _binary_labels(labels)
    probabilities = _probabilities(logits, len(labels))
    result = {}
    for label, name in ((0, "normal"), (1, "pcos")):
        values = probabilities[labels == label]
        result[f"mean_probability_{name}"] = float(values.mean(dtype=np.float32))
        result[f"std_probability_{name}"] = float(values.std(ddof=0, dtype=np.float32))
    return result


def _image_labels(frame):
    if not {"imagePath", "label"}.issubset(frame.columns):
        raise ValueError("CSV/DataFrame exige imagePath e label.")
    result = frame[["imagePath", "label"]].copy()
    if result.isna().any().any():
        raise ValueError("imagePath/label ausente.")
    result["imagePath"] = result["imagePath"].astype(str)
    if (result["imagePath"].str.strip() == "").any():
        raise ValueError("imagePath vazio.")
    if result["imagePath"].duplicated().any():
        raise ValueError("imagePath duplicado na particao.")
    _binary_labels(result["label"].to_numpy())
    result["label"] = result["label"].astype(np.int64)
    return result.sort_values("imagePath").reset_index(drop=True)


def save_augmentation_examples(img_dir, train_df, train_transform, eval_transform, out_dir):
    """Salva diagnostics/augmentation_examples.png; restaura todos os RNGs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    fig = None
    try:
        try:
            from .transforms_config import IMAGENET_MEAN, IMAGENET_STD
        except ImportError:
            from transforms_config import IMAGENET_MEAN, IMAGENET_STD

        frame = _image_labels(train_df)
        selected = pd.concat([frame[frame["label"] == label].head(2) for label in (0, 1)])
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

        def display(tensor):
            tensor = torch.as_tensor(tensor).detach().cpu()
            if tensor.ndim != 3 or tensor.shape[0] != 3 or not torch.isfinite(tensor).all():
                raise ValueError("Transform deve devolver tensor RGB CHW finito normalizado.")
            return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        from PIL import Image

        fig, axes = plt.subplots(len(selected), 5, figsize=(15, 3 * len(selected)), squeeze=False)
        titles = ["Original (treino)", "Validacao deterministica", "Aug 1", "Aug 2", "Aug 3"]
        root = Path(img_dir).resolve()
        for row_axes, (_, row) in zip(axes, selected.iterrows()):
            path = (root / row["imagePath"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Imagem de treino fora de img_dir.")
            with Image.open(path) as source:
                original = source.convert("RGB")
            validation = eval_transform(original.copy())
            repeated = eval_transform(original.copy())
            if not torch.equal(torch.as_tensor(validation), torch.as_tensor(repeated)):
                raise ValueError("eval_transform nao e deterministico.")
            images = [original, display(validation)]
            images.extend(display(train_transform(original.copy())) for _ in range(3))
            for ax, title, image in zip(row_axes, titles, images):
                ax.imshow(image)
                ax.set_title(title)
                ax.axis("off")
            name = "Normal" if row["label"] == 0 else "PCOS"
            row_axes[0].set_title(f"{name}: {row['imagePath']}\n{titles[0]}", fontsize=9)
        fig.tight_layout()
        output = Path(out_dir) / "diagnostics" / "augmentation_examples.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        return output
    finally:
        try:
            if fig is not None:
                plt.close(fig)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)


def _positive_integer(value, name):
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} deve ser inteiro positivo.")
    number = float(value)
    if not np.isfinite(number) or number < 1 or not number.is_integer():
        raise ValueError(f"{name} deve ser inteiro positivo.")
    return int(number)


def _read_config(directory, baseline, notify):
    path = directory / "config.json"
    if not path.exists() and baseline:
        notify("config.json ausente: conclusao/ausencia de debug nao verificaveis; "
               "todas as metricas do Passo 6 ficam indisponiveis.")
        return None
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuracao invalida: {path}")
    debug = config.get("debug_amostras", 0)
    if debug is None or float(debug) != 0 or config.get("debug", False):
        raise ValueError(f"Comparacao cientifica proibida para configuracao debug: {path}")
    if config.get("incomplete", False) or config.get("completed") is False:
        raise ValueError(f"Configuracao incompleta: {path}")
    if not config.get("fim"):
        raise ValueError(f"Configuracao sem fim (treino incompleto): {path}")
    epoch = _positive_integer(config.get("melhor_epoca"), "melhor_epoca")
    executed = _positive_integer(config.get("epocas_executadas"), "epocas_executadas")
    if epoch > executed:
        raise ValueError(f"melhor_epoca excede epocas_executadas: {path}")
    return config


def _read_log(directory, config, baseline, notify):
    path = directory / "logs" / "train_log.csv"
    if not path.exists() and baseline:
        notify("logs/train_log.csv ausente: metricas por epoca indisponiveis.")
        return None
    log = pd.read_csv(path)
    if log.empty or "epoch" not in log:
        raise ValueError(f"Log vazio ou sem epoch: {path}")
    epochs = pd.to_numeric(log["epoch"], errors="raise")
    if not np.isfinite(epochs).all() or (epochs % 1 != 0).any():
        raise ValueError(f"Epocas invalidas: {path}")
    log["epoch"] = epochs.astype(int)
    log = log.sort_values("epoch").reset_index(drop=True)
    expected = (int(config["epocas_executadas"]) if config is not None
                else int(log["epoch"].max()))
    if log["epoch"].tolist() != list(range(1, expected + 1)):
        raise ValueError(f"Log incompleto/duplicado ou inconsistente com config: {path}")
    return log


def _logged_value(row, columns):
    for column in columns:
        if column in row:
            value = float(row[column])
            if np.isfinite(value):
                return value
    return float("nan")


def _internal_prediction_metrics(directory, epoch, validation):
    path = directory / "diagnostics" / "val_predictions.csv"
    if not path.exists():
        path = directory / "diagnostics" / "best_validation_scores.csv"
        if not path.exists():
            return {}
    predictions = pd.read_csv(path)
    if "epoch" not in predictions:
        raise ValueError(f"Predicoes internas sem epoch: {path}")
    predictions = predictions[predictions["epoch"] == epoch].copy()
    identity = _image_labels(predictions)
    if not identity.equals(validation):
        raise ValueError(f"Predicoes nao cobrem o split interno na melhor epoca: {path}")
    labels = predictions["label"].to_numpy()
    if "logit" in predictions:
        probabilities = _probabilities(predictions["logit"].to_numpy(), len(labels))
    elif "prob_pcos" in predictions:
        probabilities = predictions["prob_pcos"].to_numpy(dtype=np.float32)
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError(f"Probabilidades internas invalidas: {path}")
    else:
        raise ValueError(f"Predicoes internas exigem logit ou prob_pcos: {path}")
    predicted = probabilities > np.float32(0.5)
    tn = int(((labels == 0) & ~predicted).sum())
    fp = int(((labels == 0) & predicted).sum())
    fn = int(((labels == 1) & ~predicted).sum())
    tp = int(((labels == 1) & predicted).sum())
    recall, specificity = tp / (tp + fn), tn / (tn + fp)
    return {
        "val_balanced_accuracy": (recall + specificity) / 2,
        "val_precision": tp / (tp + fp) if tp + fp else 0.0,
        "val_recall": recall,
        "val_specificity": specificity,
    }


def _confusion_from_logged_metrics(row, validation):
    """Recupera somente a solucao inteira unica que reproduz acc e F1 binario."""
    accuracy = _logged_value(row, ("val_acc", "val_accuracy"))
    f1 = _logged_value(row, ("val_f1",))
    if not np.isfinite(accuracy) or not np.isfinite(f1) or not 0 <= f1 <= 1:
        return {}
    n = len(validation)
    correct = round(accuracy * n)
    if not 0 <= correct <= n or not np.isclose(correct / n, accuracy, atol=1e-12, rtol=0):
        return {}
    normal = int((validation["label"] == 0).sum())
    pcos = n - normal
    candidates = []
    for tp in range(max(0, correct - normal), min(pcos, correct) + 1):
        tn = correct - tp
        fp, fn = normal - tn, pcos - tp
        denominator = 2 * tp + fp + fn
        reproduced_f1 = 2 * tp / denominator if denominator else 0.0
        if np.isclose(reproduced_f1, f1, atol=1e-12, rtol=0):
            candidates.append((tn, fp, fn, tp))
    if len(candidates) != 1:
        return {}
    tn, fp, fn, tp = candidates[0]
    recall, specificity = tp / pcos, tn / normal
    return {
        "val_balanced_accuracy": (recall + specificity) / 2,
        "val_precision": tp / (tp + fp) if tp + fp else 0.0,
        "val_recall": recall,
        "val_specificity": specificity,
    }


def _experiment_metrics(directory, config, log, validation, notify):
    values = dict.fromkeys(_METRICS, float("nan"))
    sources = {}
    if config is not None:
        epoch = int(config["melhor_epoca"])
        values["best_epoch"] = epoch
        sources["best_epoch"] = "config.json:melhor_epoca"
        values["threshold_youden"] = _logged_value(config, ("limiar_youden",))
        if np.isfinite(values["threshold_youden"]):
            sources["threshold_youden"] = "config.json:limiar_youden (validacao interna)"
        if log is not None:
            row = log.loc[log["epoch"] == epoch].iloc[0]
            for metric, columns in _METRICS.items():
                if columns:
                    values[metric] = _logged_value(row, columns)
                    if np.isfinite(values[metric]):
                        sources[metric] = f"logs/train_log.csv:epoch={epoch}"
            train_auc = _logged_value(row, ("train_auc",))
            if np.isfinite(train_auc) and np.isfinite(values["val_auc"]):
                values["train_val_auc_gap"] = train_auc - values["val_auc"]
                sources["train_val_auc_gap"] = f"logs/train_log.csv:epoch={epoch}"
            recoverable = ("val_balanced_accuracy", "val_precision", "val_recall", "val_specificity")
            if any(not np.isfinite(values[key]) for key in recoverable):
                recovered = _internal_prediction_metrics(directory, epoch, validation)
                source = f"diagnostics:predicoes internas epoch={epoch}, prob_pcos>0.5"
                if not recovered:
                    recovered = _confusion_from_logged_metrics(row, validation)
                    source = (f"logs/train_log.csv:epoch={epoch} + contagens do split; "
                              "matriz de confusao inteira unica reproduzindo acc/F1 (tolerancia 1e-12)")
                for key, value in recovered.items():
                    if not np.isfinite(values[key]):
                        values[key] = value
                        sources[key] = source
    missing = [metric for metric, value in values.items() if not np.isfinite(value)]
    if missing:
        notify("Indisponivel (NaN): " + ", ".join(missing) +
               ". Sem metrica no log/predicoes INTERNAS verificadas; "
               "sem solucao inteira unica para acc/F1; avaliacao externa nao consultada.")
    return values, {"sources": sources, "unavailable_metrics": missing}


def _plot_comparison(logs, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    try:
        for ax, (column, title) in zip(axes.flat, (
            ("train_auc", "AUC treino"), ("val_auc", "AUC validacao interna"),
            ("train_loss", "Loss treino"), ("val_loss", "Loss validacao interna"),
        )):
            unavailable = []
            for name, log in logs.items():
                if log is None or column not in log:
                    unavailable.append(name)
                    continue
                values = pd.to_numeric(log[column], errors="raise")
                ax.plot(log["epoch"], values.where(np.isfinite(values)), label=name)
                if not np.isfinite(values).all():
                    unavailable.append(f"{name} (valores ausentes)")
            if unavailable:
                ax.text(0.02, 0.02, "Indisponivel: " + ", ".join(unavailable),
                        transform=ax.transAxes, fontsize=8)
            ax.set(title=title, xlabel="Epoca", ylabel=column)
            ax.grid(alpha=0.25)
            if ax.lines:
                ax.legend()
        fig.tight_layout()
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)


def compare_with_step6(step6_dir, experiment_dir):
    """Compara melhores epocas em CSV e curvas completas; retorna o DataFrame.

    Splits diferentes, debug e configuracoes presentes incompletas geram erro
    antes de escrever saidas. Baseline sem config/log e explicitamente parcial,
    nunca tratado como comparacao cientifica verificada.
    """
    directories = {"Passo 6": Path(step6_dir), "Passo 7": Path(experiment_dir)}
    if directories["Passo 6"].resolve() == directories["Passo 7"].resolve():
        raise ValueError("Passo 6 e Passo 7 devem ser experimentos distintos.")
    messages, splits, configs, logs = [], {}, {}, {}

    def notify(name, message):
        text = f"{name}: {message}"
        messages.append(text)
        warnings.warn(text, RuntimeWarning, stacklevel=3)

    for name, directory in directories.items():
        parts = {}
        for part in ("treino", "validacao"):
            path = directory / "splits" / f"pcosgen_train_{part}.csv"
            parts[part] = _image_labels(pd.read_csv(path, dtype={"imagePath": str}))
        if set(parts["treino"]["imagePath"]) & set(parts["validacao"]["imagePath"]):
            raise ValueError(f"{name}: ha imagens comuns ao treino e validacao.")
        splits[name] = parts
    for part in ("treino", "validacao"):
        if not splits["Passo 6"][part].equals(splits["Passo 7"][part]):
            raise ValueError(f"Splits imagem-rotulo de {part} diferentes: comparacao proibida.")

    for name, directory in directories.items():
        callback = lambda message, name=name: notify(name, message)
        configs[name] = _read_config(directory, name == "Passo 6", callback)
        if configs[name] is not None:
            cfg = configs[name]
            if cfg.get("split") != "grupo" or cfg.get("seed") != 42:
                raise ValueError(f"{name}: comparacao exige split grupo e seed 42.")
            if cfg.get("metrics_threshold", 0.5) != 0.5:
                raise ValueError(f"{name}: comparacao exige metricas por epoca no limiar 0.5.")
        logs[name] = _read_log(directory, configs[name], name == "Passo 6", callback)

    data, availability = {}, {}
    for name, directory in directories.items():
        data[name], availability[name] = _experiment_metrics(
            directory, configs[name], logs[name], splits[name]["validacao"],
            lambda message, name=name: notify(name, message),
        )
        for column in ("train_auc", "val_auc", "train_loss", "val_loss"):
            log = logs[name]
            if log is None or column not in log or not np.isfinite(
                pd.to_numeric(log[column], errors="raise")
            ).all():
                notify(name, f"Curva {column} ausente ou parcial.")
    table = pd.DataFrame({"metric": list(_METRICS), **{
        name: [values[metric] for metric in _METRICS] for name, values in data.items()
    }})
    output = directories["Passo 7"] / "comparacao"
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "comparacao_passo6_passo7.csv", index=False, na_rep="NaN")
    _plot_comparison(logs, output / "curvas_passo6_passo7.png")
    report = {
        "identical_image_label_splits": True,
        "scientific_comparison_verified": all(config is not None for config in configs.values())
        and all(log is not None for log in logs.values()),
        "complete_metrics": all(not info["unavailable_metrics"] for info in availability.values()),
        "classification_threshold": 0.5,
        "classification_rule": "prob_pcos > 0.5",
        "external_evaluation_used": False,
        "experiments": availability,
        "warnings": messages,
    }
    with (output / "disponibilidade_passo6_passo7.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Comparacao {'parcial' if messages else 'completa'} salva em {output}; "
          "apenas treino/validacao interna.")
    return table


def main():
    parser = argparse.ArgumentParser(description="Compara Passo 6/7 sem avaliacao externa.")
    parser.add_argument("--passo6", type=Path, required=True)
    parser.add_argument("--passo7", type=Path, required=True)
    args = parser.parse_args()
    try:
        compare_with_step6(args.passo6, args.passo7)
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.exit(1, f"Comparacao proibida/indisponivel: {error}\n")


if __name__ == "__main__":
    main()
