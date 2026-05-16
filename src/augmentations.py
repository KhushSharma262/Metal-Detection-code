import torch
import torchvision.transforms as T
from src.config import cfg

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
aug = cfg.augmentation


class AddGaussianNoise:
    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return (tensor + noise).clamp(0.0, 1.0)


def get_train_transforms():
    size  = cfg.dataset.image_size
    scale = tuple(cfg.augmentation.random_resized_crop_scale)
    return T.Compose([
        T.Resize((size, size)),
        T.RandomResizedCrop(size=size, scale=scale,
                            interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=aug.random_rotation_degrees),
        T.RandomPerspective(distortion_scale=aug.perspective_distortion, p=0.4),
        T.ColorJitter(brightness=aug.brightness, contrast=aug.contrast,
                      saturation=aug.saturation, hue=aug.hue),
        T.GaussianBlur(kernel_size=aug.gaussian_blur_kernel, sigma=(0.1, 1.5)),
        T.ToTensor(),
        AddGaussianNoise(std=aug.gaussian_noise_std),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms():
    size = cfg.dataset.image_size
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inverse_transform():
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    class Denormalize:
        def __call__(self, tensor):
            return (tensor * std + mean).clamp(0.0, 1.0)

    return Denormalize()
