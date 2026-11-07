"""
transforms_config.py
Pipeline de pre-processamento e data augmentation para o projeto.

Treino: redimensionamento + augmentation + normalizacao
Val/Test: redimensionamento + normalizacao (sem augmentation)

Versao 2: removido ColorJitter (ultrassom e' monocromatico, jitter de cor
adiciona ruido sem ganho semantico). Mantido flip horizontal e rotacao leve.
Usada no experimento de referencia (outputs/models/best_model.pt).

Versao 3 (PR #1 do Luan): ColorJitter leve e RandomAffine (so' translacao de
ate 5%) no treino. As transformacoes sao aplicadas em memoria a cada epoca;
os arquivos de imagem nao sao alterados. O transform de avaliacao nao mudou.

Versao 4 (relatorio tecnico do Luan, secao 4.1), escolhida por opcao para
permitir a ablacao passo a passo:
- preproc="rgb_resize" (padrao): Resize((224, 224)) em RGB, como antes.
- preproc="gray_pad": escala de cinza replicada nos 3 canais + PadToSquare
  (preserva a proporcao), no treino e na avaliacao.
- augment="pr1" (padrao): augmentation do PR #1.
- augment="ultrassom": RandomResizedCrop, flip, RandomAffine, brilho/contraste,
  GaussianBlur, ruido speckle e RandomErasing (exige preproc="gray_pad").
- augment="ultrassom_moderado": conservador, sem crop, blur ou erasing
  (exige preproc="gray_pad").
"""

import random

import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224

PREPROCS = ("rgb_resize", "gray_pad")
AUGMENTS = ("pr1", "ultrassom", "ultrassom_moderado")


class PadToSquare:
    """Completa a imagem com preto ate ficar quadrada (preserva proporcao)."""

    def __call__(self, img):
        w, h = img.size
        side = max(w, h)
        canvas = Image.new(img.mode, (side, side), 0)
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas

    def __repr__(self):
        return "PadToSquare()"


class SpeckleNoise:
    """Ruido multiplicativo tipico de ultrassom, igual nos 3 canais."""

    def __init__(self, std=0.1, p=0.3):
        self.std = std
        self.p = p

    def __call__(self, t):
        if random.random() >= self.p:
            return t
        noise = torch.randn_like(t[:1]) * self.std
        return (t * (1 + noise)).clamp(0, 1)

    def __repr__(self):
        return f"SpeckleNoise(std={self.std}, p={self.p})"


def _preproc_steps(preproc):
    if preproc == "rgb_resize":
        return [transforms.Resize((INPUT_SIZE, INPUT_SIZE))]
    if preproc == "gray_pad":
        return [
            transforms.Grayscale(num_output_channels=3),
            PadToSquare(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        ]
    raise ValueError(f"preproc invalido: {preproc} (opcoes: {PREPROCS})")


def get_train_transform(preproc="rgb_resize", augment="pr1"):
    if augment == "pr1":
        return transforms.Compose(_preproc_steps(preproc) + [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    if augment == "ultrassom":
        if preproc != "gray_pad":
            raise ValueError("augment='ultrassom' exige preproc='gray_pad'")
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            PadToSquare(),
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
            transforms.ToTensor(),
            SpeckleNoise(std=0.1, p=0.3),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
        ])
    if augment == "ultrassom_moderado":
        if preproc != "gray_pad":
            raise ValueError("augment='ultrassom_moderado' exige preproc='gray_pad'")
        return transforms.Compose(_preproc_steps(preproc) + [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            SpeckleNoise(std=0.05, p=0.20),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    raise ValueError(f"augment invalido: {augment} (opcoes: {AUGMENTS})")


def get_eval_transform(preproc="rgb_resize"):
    return transforms.Compose(_preproc_steps(preproc) + [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


if __name__ == "__main__":
    img_dummy = Image.new("RGB", (300, 280), color="gray")
    for preproc, augment in [
        ("rgb_resize", "pr1"),
        ("gray_pad", "pr1"),
        ("gray_pad", "ultrassom"),
        ("gray_pad", "ultrassom_moderado"),
    ]:
        out_train = get_train_transform(preproc, augment)(img_dummy)
        out_eval = get_eval_transform(preproc)(img_dummy)
        print(f"{preproc}/{augment}: train={tuple(out_train.shape)} eval={tuple(out_eval.shape)}")
    print("Pipeline funcionando corretamente.")
