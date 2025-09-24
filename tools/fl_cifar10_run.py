import torch
from torch import nn
from torchvision import transforms, datasets
import numpy as np
import copy
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
import logging
import wandb

from src.models.resnet import ResNet18, ResidualBlock
from src.trainers.fl_client_good import FLClient
from src.datasets.fl_dataset import DatasetSplit, dataset_iid, cifar_user_dataset, create_transforms, dirichlet_distribution_dict_users
from src.utils.utils import set_seed, AverageMeter, setup_logging_with_color, prRed, prGreen, setup_logging_color_message_only
from src.federated_algorithms.fedavg import FedAvg
from src.trainers.fl_server import FLServerBase


@hydra.main(version_base=None, config_path="../configs", config_name="fl_config")
def main(cfg: DictConfig) -> None:
    setup_logging_color_message_only(file_name="train_fl_cifar10.log", directory=cfg.log_dir)
    logger = logging.getLogger(__name__)
    set_seed()
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    prGreen("Start: FL ResNet18 on CIFAR10", logger=logger) # this is to identify the program in the slurm outputs files

    # configs
    device = torch.device(f'cuda:{cfg.device}' if torch.cuda.is_available() else 'cpu')
    num_users = cfg.server.num_users
    global_epochs = cfg.server.global_epochs
    participation_rate = cfg.server.participation_rate # 1이면 full participation, 1보다 작으면 partial participation

    # wandb setup
    distributed_method = "fl"
    algorithm = cfg.algorithm.name
    model_name = "resnet18"
    dataset_name = "cifar10"
    lr = cfg.client.optimizer.lr
    global_epochs = cfg.server.global_epochs
    etc = f"beta={cfg.dataset.heterogeneity.beta}"
    etc += f"_client_num={num_users}"
    if algorithm == "fedprox":
        etc += f"_mu={cfg.algorithm.mu}"

    project_name = "FL ResNet18 on CIFAR10"
    exp_name = f"{distributed_method}_{algorithm}_{etc}_{model_name}_{dataset_name}_lr{lr}_bs{cfg.client.train.batch_size}_globalep{global_epochs}_localep{cfg.client.train.local_epochs}"
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    
    # Dataset preparation ==========================================================
    # 데이터 전처리 및 향상
    transform_train = create_transforms(cfg.dataset.train.transforms)
    transform_test = create_transforms(cfg.dataset.test.transforms)
    dataset_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    dataset_test = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)

    # cifar10 train dataset(label)
    targets = dataset_train.targets
    logger.info(f"cifar10 train dataset(label) length: {len(targets)}")
    logger.info(f"cifar10 train dataset(label) unique: {np.unique(targets)}")

    # cifar10 train dataset (dirichlet distribution)
    dict_users_train = dirichlet_distribution_dict_users(targets, num_users, alpha=cfg.dataset.heterogeneity.beta, min_size=10)
    # cifar10 test dataset (iid distribution)
    dict_users_test = dataset_iid(dataset_test, num_users)

    # -----------------------------------------------
    # dict_users=cifar_user_dataset(dataset_train,num_users,0)
    # cifar10_dirichlet_0_1 = Path('datasets/cifar0.1.txt')
    # dict_users_train = eval(cifar10_dirichlet_0_1.read_text())
    # dict_users_test = dataset_iid(dataset_test, num_users)

    total_items_count = 0
    logger.info("dict_users with dirichlet distribution")
    for idx in range(len(dict_users_train)):
        logger.info(f"\t\tdict_users[{idx}] length: {len(dict_users_train[idx])}")
        total_items_count += len(dict_users_train[idx])
    logger.info(f"total_items_count: {total_items_count}")
    

    # Model definition ==========================================================
    model = ResNet18(ResidualBlock)
    model.to(device)

    with wandb.init(project=project_name, config=cfg_dict, name=exp_name) as run:
        # Client collection ==========================================================
        clients = []
        for idx in range(num_users):
            client = FLClient(
                cfg=cfg.client,
                idx=idx,
                device=device,
                model=copy.deepcopy(model).to(device),
                dataset_train=dataset_train,
                dataset_test=dataset_test,
                dataset_split_dict_train=dict_users_train[idx],
                dataset_split_dict_test=dict_users_test[idx]
            )
            clients.append(client)
        
        server = FLServerBase(
            cfg=cfg.server,
            logger=logger,
            wandb=run,
            device=device,
            clients=clients,
            global_params_dict=model.state_dict(keep_vars=True)
        )
        
        # Training/Testing simulation ==========================================================
        server.train()
        
        #===============================================================================
        logger.info(f"loss_train_collect: {server.global_loss_train_collect}")
        logger.info(f"loss_test_collect: {server.global_loss_test_collect}")
        logger.info(f"acc_train_collect: {server.global_acc_train_collect}")
        logger.info(f"acc_test_collect: {server.global_acc_test_collect}")

        prGreen(f"best test acc: {max(server.global_acc_test_collect)}", logger=logger)

    #=============================================================================
    #                         Program Completed
    #============================================================================= 

if __name__ == "__main__":
    main()