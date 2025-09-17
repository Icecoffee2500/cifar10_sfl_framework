#=============================================================================
# Split learning: ResNet18 on HAM10000
# HAM10000 dataset: Tschandl, P.: The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions (2018), doi:10.7910/DVN/DBW86T

# We have three versions of our implementations
# Version1: without using socket and no DP+PixelDP
# Version2: with using socket but no DP+PixelDP
# Version3: without using socket but with DP+PixelDP

# This program is Version1: Single program simulation 
# ============================================================================
import torch
from torch import nn
from torchvision import transforms, datasets
import numpy as np
import copy
from pathlib import Path

from models.resnet_client_side import ResNet18_client_side
from models.resnet_server_side import ResNet18_server_side
from models.residual_block import ResidualBlock
from datasets.fl_dataset import dataset_iid
from trainers.sl_client import Client
from trainers.sl_server import SLServer
from utils.utils import set_seed
import logging
from utils.utils import setup_logging_color_message_only, prGreen, prRed
from datasets.fl_dataset import dirichlet_distribution_dict_users


setup_logging_color_message_only(file_name="train_sl_cifar10.log")
logger = logging.getLogger(__name__)
set_seed()

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

#===================================================================  
# program = "SL ResNet18 on CIFAR10"
# print(f"---------{program}----------")              # this is to identify the program in the slurm outputs files
prGreen("Start: SL ResNet18 on CIFAR10", logger=logger)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')


#===================================================================  
# No. of users
num_users = 10
# num_users = 100
epochs = 200
frac = 1   # participation of clients; if 1 then 100% clients participate in SL
lr = 0.01

#=====================================================================================================
#                           Client-side Model definition
#=====================================================================================================
# Model at client side
net_glob_client = ResNet18_client_side(ResidualBlock)
# if torch.cuda.device_count() > 1:
#     print("We use", torch.cuda.device_count(), "GPUs")
#     net_glob_client = nn.DataParallel(
#         net_glob_client)  # to use the multiple GPUs; later we can change this to CPUs only

net_glob_client.to(device)


# =====================================================================================================
#                           Server-side Model definition
# =====================================================================================================
# Model at server side
net_glob_server = ResNet18_server_side(ResidualBlock)  # 7 is my numbr of classes
# if torch.cuda.device_count() > 1:
#     print("We use", torch.cuda.device_count(), "GPUs")
#     net_glob_server = nn.DataParallel(net_glob_server)  # to use the multiple GPUs

net_glob_server.to(device)

#===================================================================================
# For Server Side Loss and Accuracy 
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []

#=============================================================================
#                         Data loading 
#============================================================================= 

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

# Data preprocessing and augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
dataset_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
dataset_test = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
# dataset_train = SkinData(train, transform = train_transforms)
# dataset_test = SkinData(test, transform = test_transforms)

# cifar10 train dataset(label)
targets = dataset_train.targets
logger.info(f"cifar10 train dataset(label) length: {len(targets)}")
logger.info(f"cifar10 train dataset(label) unique: {np.unique(targets)}")

# cifar10 train dataset (dirichlet distribution)
dict_users_train = dirichlet_distribution_dict_users(targets, num_users, alpha=0.1, min_size=10)
# cifar10 test dataset (iid distribution)
dict_users_test = dataset_iid(dataset_test, num_users)

# ----------------------------------------------------------------
# with open('beta=0.1.pkl', 'rb') as file:
#     dict_users=pickle.load(file)
# dict_users=cifar_user_dataset(dataset_train,num_users,0)
# ----------------------------------------------------------------
# cifar10_dirichlet_0_1 = Path('datasets/cifar0.1.txt')
# dict_users_train = eval(cifar10_dirichlet_0_1.read_text())
# dict_users_test = dataset_iid(dataset_test, num_users)

total_items_count = 0
logger.info("dict_users with dirichlet distribution")
for idx in range(len(dict_users_train)):
    logger.info(f"\t\tdict_users[{idx}] length: {len(dict_users_train[idx])}")
    total_items_count += len(dict_users_train[idx])
logger.info(f"total_items_count: {total_items_count}")

server = SLServer(
    net_glob_server=net_glob_server,
    criterion=nn.CrossEntropyLoss(),
    device=device,
    num_users=num_users
)

clients = []
for idx in range(num_users):
    client = Client(
        net_glob_client,
        server,
        idx,
        lr,
        device,
        frac=frac,
        total_num_users=num_users,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        idxs=dict_users_train[idx],
        idxs_test=dict_users_test[idx]
    )
    clients.append(client)

#net_glob_client.train()
# this epoch is global epoch, also known as rounds
for iter in range(epochs):
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace = False)

    # Sequential training/testing among clients      
    for idx in idxs_users:
        local = clients[idx]
        # Training ------------------
        w_client = local.train(net = copy.deepcopy(net_glob_client).to(device))
              
        # Testing -------------------
        local.evaluate(net=copy.deepcopy(net_glob_client).to(device), ell=iter)
        
        # copy weight to net_glob_client -- use to update the client-side model of the next client to be trained
        net_glob_client.load_state_dict(w_client)
   
#===================================================================================     

prGreen("Training and Evaluation completed!", logger=logger)    

#===============================================================================
# Save output data to .excel file (we use for comparision plots)
# round_process = [i for i in range(1, len(local.server.acc_train_collect)+1)]

logger.info(f"loss_train_collect: {server.loss_train_collect}")
logger.info(f"loss_test_collect: {server.loss_test_collect}")
logger.info(f"acc_train_collect: {server.acc_train_collect}")
logger.info(f"acc_test_collect: {server.acc_test_collect}")

prGreen(f"best test acc: {max(server.acc_test_collect)}", logger=logger)

#=============================================================================
#                         Program Completed
#============================================================================= 



 







