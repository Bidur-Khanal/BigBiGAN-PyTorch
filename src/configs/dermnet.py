from src.configs import hparams
from src.configs import dataset_configs
from src.configs import utilities

config = {
    "ds_name": "dermnet",
    "num_cls": 0,
    "loading_normalization_mean": [0.485, 0.456, 0.406],
    "loading_normalization_var": [0.229, 0.224, 0.225],
    "w_init": None, # torch.nn.init.orthogonal_,
    "save_metric_interval": 10,
    "logging_interval": 35,
    "enc_model":"resnet_18",
    "epochs": 400,
    "save_model_interval": 20,
    **hparams.hparams,
    **dataset_configs.img_color_128x128_config,
}

config = utilities.Config(**config)
