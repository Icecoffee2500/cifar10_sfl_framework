#===========================================================
# Federated learning: ResNet18 on HAM10000
# HAM10000 dataset: Tschandl, P.: The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions (2018), doi:10.7910/DVN/DBW86T

# We have three versions of our implementations
# Version1: without using socket and no DP+PixelDP
# Version2: with using socket but no DP+PixelDP
# Version3: without using socket but with DP+PixelDP

# This program is Version1: Single program simulation 
# ===========================================================
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

from models.resnet import ResNet18, ResidualBlock
from trainers.fl_local_update import FLClient
from datasets.fl_dataset import DatasetSplit, dataset_iid, cifar_user_dataset, create_transforms, dirichlet_distribution_dict_users
from utils.utils import set_seed, AverageMeter, setup_logging_with_color, prRed, prGreen, setup_logging_color_message_only
from federated_algorithms.fedavg import FedAvg


@hydra.main(version_base=None, config_path="../configs", config_name="fl_config")
def main(cfg: DictConfig) -> None:
    # setup_logging(file_name="train_fl_cifar10.log")
    # setup_logging_with_color(file_name="train_fl_cifar10.log")
    setup_logging_color_message_only(file_name="train_fl_cifar10.log")
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
    frac = cfg.server.frac # 1이면 full participation, 1보다 작으면 partial participation

    # wandb setup
    algorithm = "fl"
    model_name = "resnet18"
    dataset_name = "cifar10"
    lr = cfg.client.optimizer.lr
    global_epochs = cfg.server.global_epochs
    etc = f"beta={cfg.dataset.heterogeneity.beta}"
    etc += f"_client_num={num_users}"

    project_name = "FL ResNet18 on CIFAR10"
    exp_name = f"{algorithm}_{model_name}_{dataset_name}_lr{lr}_bs{cfg.client.train.batch_size}_globalep{global_epochs}_localep{cfg.client.train.local_epochs}_{etc}"
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
    # with open('beta=0.1.pkl', 'rb') as file:
    #     dict_users=pickle.load(file)
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
    model_global = ResNet18(ResidualBlock)
    # if torch.cuda.device_count() > 1:
    #     print("We use",torch.cuda.device_count(), "GPUs")
    #     model_global = nn.DataParallel(model_global)   # to use the multiple GPUs 

    model_global.to(device)
    model_global.train()
    # copy weights
    global_weight = model_global.state_dict()

    # Loss and Accuracy collection ==========================================================
    loss_train_collect = []
    acc_train_collect = []
    loss_test_collect = []
    acc_test_collect = []

    # Client collection ==========================================================
    clients = []
    for idx in range(num_users):
        client = FLClient(
            cfg=cfg.client,
            idx=idx,
            device=device,
            model=copy.deepcopy(model_global).to(device),
            dataset_train=dataset_train,
            dataset_test=dataset_test,
            dataset_split_dict_train=dict_users_train[idx],
            dataset_split_dict_test=dict_users_test[idx]
        )
        clients.append(client)
    
    local_weights = []
    local_train_losses = AverageMeter()
    local_train_accuracies = AverageMeter()
    local_test_losses = AverageMeter()
    local_test_accuracies = AverageMeter()

    with wandb.init(project=project_name, config=cfg_dict, name=exp_name) as run:
        # Training/Testing simulation ==========================================================
        for global_epoch in range(global_epochs):
            # w_locals, loss_locals_train, acc_locals_train, loss_locals_test, acc_locals_test = [], [], [], [], []
            local_weights = []
            local_train_losses.reset()
            local_train_accuracies.reset()
            local_test_losses.reset()
            local_test_accuracies.reset()

            wdb_log_dict = {}

            # 참여하는 client 선택 # 매 round마다 참여하는 client가 달라짐.
            num_users_participated = max(int(frac * num_users), 1) # full participation과 partial participation을 처리하기 위해서
            idxs_users = np.random.choice(range(num_users), num_users_participated, replace = False) # partial participation일 때는 전체 client 중에서 참여하는 client를 무작위로 고름.
            
            # Training/Testing simulation
            for idx in idxs_users: # each client
                # client 선택
                client = clients[idx]
                # Training ------------------
                local_weight, loss_train, acc_train = client.train()

                local_weights.append(copy.deepcopy(local_weight))

                local_train_losses.update(loss_train)
                local_train_accuracies.update(acc_train)
                wdb_log_dict[f"client_{idx}/train_loss"] = loss_train # wandb log
                wdb_log_dict[f"client_{idx}/train_acc"] = acc_train # wandb log
                
                # Testing -------------------
                loss_test, acc_test = client.evaluate(global_model=model_global)

                local_test_losses.update(loss_test)
                local_test_accuracies.update(acc_test)
                wdb_log_dict[f"client_{idx}/test_loss"] = loss_test # wandb log
                wdb_log_dict[f"client_{idx}/test_acc"] = acc_test # wandb log

            # Federation process ==========================================================
            global_weight = FedAvg(local_weights)
            prGreen("------------------------------------------------", logger=logger)
            prGreen("------ Federation process at Server-Side -------", logger=logger)
            prGreen("------------------------------------------------", logger=logger)
            
            # update global model ==========================================================
            # copy weight to global model and distribute the model to all users
            model_global.load_state_dict(global_weight)
            for client in clients:
                client.model.load_state_dict(global_weight)
            
            # Save Train/Test accuracy ==========================================================
            acc_avg_train = local_train_accuracies.avg
            acc_train_collect.append(acc_avg_train)
            wdb_log_dict[f"train/acc_avg"] = acc_avg_train # wandb log

            acc_avg_test = local_test_accuracies.avg
            acc_test_collect.append(acc_avg_test)
            wdb_log_dict[f"test/acc_avg"] = acc_avg_test # wandb log

            # Save Train/Test loss ==========================================================
            loss_avg_train = local_train_losses.avg
            loss_train_collect.append(loss_avg_train)
            wdb_log_dict[f"train/loss_avg"] = loss_avg_train # wandb log

            loss_avg_test = local_test_losses.avg
            loss_test_collect.append(loss_avg_test)
            wdb_log_dict[f"test/loss_avg"] = loss_avg_test # wandb log
            
            
            # Print results ==========================================================
            prGreen('------------------- SERVER ----------------------------------------------', logger=logger)
            prGreen(f"Train: Round {global_epoch:3d}, Avg Accuracy {acc_avg_train:.3f} | Avg Loss {loss_avg_train:.3f}", logger=logger)
            prGreen(f"Test:  Round {global_epoch:3d}, Avg Accuracy {acc_avg_test:.3f} | Avg Loss {loss_avg_test:.3f}", logger=logger)
            prGreen('-------------------------------------------------------------------------', logger=logger)

            run.log(wdb_log_dict, step=global_epoch)

        prRed("Training and Evaluation completed!", logger=logger)    

        #===============================================================================
        # Save output data to .excel file (we use for comparision plots)
        # round_process = [i for i in range(1, len(acc_train_collect)+1)]
        # df = DataFrame({'round': round_process,'acc_train':acc_train_collect, 'acc_test':acc_test_collect})
        # file_name = program+".xlsx"
        # df.to_excel(file_name, sheet_name= "v1_test", index = False)
        logger.info(f"loss_train_collect: {loss_train_collect}")
        logger.info(f"loss_test_collect: {loss_test_collect}")
        logger.info(f"acc_train_collect: {acc_train_collect}")
        logger.info(f"acc_test_collect: {acc_test_collect}")

        prGreen(f"best test acc: {max(acc_test_collect)}", logger=logger)

    #=============================================================================
    #                         Program Completed
    #============================================================================= 

if __name__ == "__main__":
    main()