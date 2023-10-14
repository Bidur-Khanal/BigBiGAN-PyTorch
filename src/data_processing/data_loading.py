from torchvision.datasets import FashionMNIST, MNIST
from torchvision import datasets, transforms
from torch.utils import data as tdataset

from src.data_processing import datasets as mydatasets

def get_dataloader(dataset, bs):
    loader = tdataset.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        drop_last=True,
    )
    return loader


def get_CIFAR10_loader(data_path, config):
    dataset = datasets.CIFAR10(
        root=data_path,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
        ])
    )
    return get_dataloader(dataset, config.bs)


def get_CIFAR100_loader(data_path, config):
    dataset = datasets.CIFAR100(
        root=data_path,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
        ])
    )
    return get_dataloader(dataset, config.bs)


def get_imagenette_loader(data_path, config): # shorter edge 160
    dataset = mydatasets.Imagenette(
        root=data_path,
        transform=transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
        ])
    )
    return get_dataloader(dataset, config.bs)


def get_imagewoof_loader(data_path, config): # shorter edge 160
    dataset = mydatasets.Imagenette(
        root=data_path,
        csv="noisy_imagewoof.csv",
        transform=transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
        ])
    )
    return get_dataloader(dataset, config.bs)


def get_FMNIST_loader(data_path, config):
    dataset = FashionMNIST(
        data_path,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
        ])
    )
    return get_dataloader(dataset, config.bs)


def get_MNIST_loader(data_path, config):
    dataset = MNIST(
        data_path,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
        ])
    )
    return get_dataloader(dataset, config.bs)


def get_Fetal_loader(data_path, config):
    dataset = mydatasets.custom_FETAL_PLANE_faster(data_path,train=True, transform=transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.loading_normalization_mean,std=config.loading_normalization_var),
        ]))
    
    return get_dataloader(dataset, config.bs)

def get_Covid19_xray_loader(data_path, config):
    dataset = mydatasets.custom_COVID19_Xray_faster(data_path,train=True, transform=transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.loading_normalization_mean,std=config.loading_normalization_var),
        ]))
    
    return get_dataloader(dataset, config.bs)

def get_Histopathology_loader(data_path, config):
    dataset = mydatasets.custom_histopathology_faster(data_path,train=True, transform=transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.loading_normalization_mean,std=config.loading_normalization_var),
        ]))
    
    return get_dataloader(dataset, config.bs)

def get_Mura_loader(data_path, config):
    dataset = mydatasets.MURA_faster(data_path,train=True, transform=transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.loading_normalization_mean,std=config.loading_normalization_var),
        ]))
    
    return get_dataloader(dataset, config.bs)

def get_Dermnet_loader(data_path, config):
    dataset = mydatasets.custom_dermnet_faster(data_path,train=True, transform=transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.loading_normalization_mean,std=config.loading_normalization_var),
        ]))
    
    return get_dataloader(dataset, config.bs)

loaders = {
    "MNIST": get_MNIST_loader,
    "FMNIST": get_FMNIST_loader,
    "CIFAR10": get_CIFAR10_loader,
    "CIFAR100": get_CIFAR100_loader,
    "imagenette": get_imagenette_loader,
    "imagewoof": get_imagewoof_loader,
    "fetal":get_Fetal_loader,
    "covid19_xray":get_Covid19_xray_loader,
    "histopathology":get_Histopathology_loader,
    "mura":get_Mura_loader,
    "dermnet":get_Dermnet_loader,
}


def get_supported_loader(name):
    return loaders[name]
