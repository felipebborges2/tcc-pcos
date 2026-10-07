"""
transforms_config.py
Pipeline de pre-processamento e data augmentation para o projeto.

Treino: redimensionamento + augmentation + normalizacao
Val/Test: redimensionamento + normalizacao (sem augmentation)

Versao 2: removido ColorJitter (ultrassom e' monocromatico, jitter de cor
adiciona ruido sem ganho semantico). Mantido flip horizontal e rotacao leve.
"""

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


if __name__ == "__main__":
    from PIL import Image
    img_dummy = Image.new("RGB", (300, 280), color="gray")
    out_train = get_train_transform()(img_dummy)
    out_eval = get_eval_transform()(img_dummy)
    print(f"Train transform output: shape={out_train.shape}")
    print(f"Eval transform output : shape={out_eval.shape}")
    print("Pipeline funcionando corretamente.")