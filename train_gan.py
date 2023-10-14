import argparse
import itertools
import torch
from src.pipeline import pipeline
from src.training_utils import training_utils
import neptune as neptune
from neptune.types import File

EXP_HPARAMS = {
    "params": (
        {},
    ),
    "seeds": (420,),
}



def run_experiments(args,neptune_run=None):
    for hparams_overwrite_list, seed in itertools.product(EXP_HPARAMS["params"], EXP_HPARAMS["seeds"]):
        config = training_utils.get_config(args.dataset)
        hparams_str = ""
        for k, v in hparams_overwrite_list.items():
            config[k] = v
            hparams_str += str(k) + "-" + str(v) + "_"
        config["model_architecture"] = args.model_architecture
        config["hparams_str"] = hparams_str.strip("_")
        config["seed"] = seed
        run_experiment(config,args, neptune_run)


def run_experiment(config,args,neptune_run= None):
    training_utils.set_random_seed(seed=config.seed, device=config.device)
    if args.model_architecture == "bigbigan":
        training_pipeline = pipeline.BigBiGANPipeline.from_config(data_path=args.data_path, config=config)
    elif args.model_architecture == "bigbiwgan":
        training_pipeline = pipeline.BigBiWGANPipeline.from_config(data_path=args.data_path, config=config)
    elif args.model_architecture == "biggan":
        training_pipeline = pipeline.GANPipeline.from_config(data_path=args.data_path, config=config)
    else:
        raise ValueError(f"Architecture type {args.model_architecture} is not supported")
    training_pipeline.train_model(neptune_run)


def main(args, neptune_run = None):

    run_experiments(args, neptune_run)



if __name__ == "__main__":

    run = neptune.init_run(
    project="bidur/BigBiGAN",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxM2NkY2I5MC01OGUzLTQzZWEtODYzYi01YTZiYmFjZmM4NmIifQ==",
)  # your credentials
    

    parser = argparse.ArgumentParser(description='PyTorch CNN Training')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FMNIST",
                        choices=["FMNIST", "MNIST", "CIFAR10", "CIFAR100", "imagenette", "imagewoof", "fetal", "histopathology", "covid19_xray","dermnet","mura"], help="dataset name")
    parser.add_argument("--data_path", type=str, default="../input/fmnist-dataset",
                        help="path to dataset root folder")
    parser.add_argument("--model_architecture", type=str, default="bigbigan",
                        choices=["bigbigan", "biggan"], help="type of architecture used in training")
    args = parser.parse_args()



    args = parser.parse_args()
    params = vars(args)
    run["parameters"] = params
    run["main_file"].upload_files("*.py")
    run["src/configs"].upload_files("src/configs/*.py")
    run["src/data_processing"].upload_files("src/data_processing/*.py")
    run["src/model"].upload_files("src/model/*.py")
    run["src/pipeline"].upload_files("src/pipeline/*.py")
    run["src/train_utils"].upload_files("src/training_utils/*.py")

        
    main(args, run)