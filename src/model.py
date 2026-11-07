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


STAGES = ["layer4", "layer3", "layer2", "layer1"]  # do mais profundo ao mais raso


def set_trainable_stages(model, n_stages):
    """
    Congela o backbone e libera a cabeca (fc) + os n_stages mais profundos.
    n_stages=0 -> so a cabeca; n_stages=4 -> rede inteira (exceto conv1/bn1).
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    for name in STAGES[:n_stages]:
        for p in getattr(model, name).parameters():
            p.requires_grad = True


def freeze_batchnorm_stats(model):
    """Congela estatisticas e affine de BN; chamar apos train() e liberar stages."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            if module.weight is not None:
                module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False


def get_param_groups(model, lr_backbone, lr_head, weight_decay=None):
    """Separa backbone/cabeca, opcionalmente sem decay em biases e parametros 1D.

    Inclui parametros congelados para permitir o descongelamento gradual.
    weight_decay=None preserva os dois grupos originais.
    """
    head_ids = {id(p) for p in model.fc.parameters()}
    if weight_decay is not None:
        groups = {
            f"{role}_{decay}": {
                "params": [],
                "name": f"{role}_{decay}",
                "role": role,
                "lr": lr,
                "weight_decay": weight_decay if decay == "decay" else 0.0,
            }
            for role, lr in [("backbone", lr_backbone), ("head", lr_head)]
            for decay in ["decay", "no_decay"]
        }
        for name, param in model.named_parameters():
            role = "head" if id(param) in head_ids else "backbone"
            decay = "no_decay" if name == "bias" or name.endswith(".bias") or param.ndim == 1 else "decay"
            groups[f"{role}_{decay}"]["params"].append(param)
        return list(groups.values())

    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return [
        {"params": backbone, "lr": lr_backbone},
        {"params": list(model.fc.parameters()), "lr": lr_head},
    ]


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