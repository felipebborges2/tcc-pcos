"""
model.py
Constroi a ResNet-50 com transfer learning para classificacao binaria.

- Pesos pre-treinados no ImageNet
- Camada final substituida por Dropout + Linear para classificacao binaria
- Fine-tuning completo (todas as camadas treinaveis)

Versao 2: adicionado Dropout(0.5) antes da camada final para regularizacao.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_resnet50(pretrained=True, freeze_backbone=False, dropout=0.5):
    """
    Constroi uma ResNet-50 adaptada para classificacao binaria.

    Args:
        pretrained: se True, usa pesos pre-treinados no ImageNet.
        freeze_backbone: se True, congela todas as camadas exceto a final.
        dropout: probabilidade de dropout antes da camada final.

    Returns:
        modelo nn.Module pronto para treino.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 1),
    )

    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    print("Construindo ResNet-50 com pesos do ImageNet (dropout=0.5)...")
    model = build_resnet50(pretrained=True, freeze_backbone=False)

    total, trainable = count_parameters(model)
    print(f"Parametros totais    : {total:,}")
    print(f"Parametros treinaveis: {trainable:,}")

    print("\nTeste com tensor dummy...")
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input  shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    if torch.cuda.is_available():
        print(f"\nGPU disponivel: {torch.cuda.get_device_name(0)}")

    print("\nModelo construido com sucesso.")