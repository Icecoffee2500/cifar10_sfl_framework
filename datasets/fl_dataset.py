import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import yaml
from omegaconf import OmegaConf
from torchvision import transforms

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

import logging
logger = logging.getLogger(__name__)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

# Custom dataset prepration in Pytorch format
class SkinData(Dataset):
    def __init__(self, df, transform = None):
        
        self.df = df
        self.transform = transform
        
    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, index):
        
        X = Image.open(self.df['path'][index]).resize((64, 64))
        y = torch.tensor(int(self.df['target'][index]))
        
        if self.transform:
            X = self.transform(X)
        
        return X, y

def create_transforms(cfg_transforms):
    # logger.info(f"cfg_transforms:\n{yaml.dump(OmegaConf.to_container(cfg_transforms))}")
    transform_list = []
    for transform in cfg_transforms:
        transform_type = transform.type
        params = {k: v for k, v in transform.items() if k != 'type'} # type 제외한 나머지 파라미터
        transform_class = getattr(transforms, transform_type) # (예시) RandomCrop -> transforms.RandomCrop 이런 식으로 변환
        if params:
            transform_list.append(transform_class(**params)) # (예시) RandomCrop(size=32, padding=4) 이런 식으로 변환
        else:
            transform_list.append(transform_class()) # (예시) RandomCrop() 이런 식으로 변환
    # print(f"transform_list: {transform_list}")
    return transforms.Compose(transform_list)

#=====================================================================================================
# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
# IID HAM10000 datasets will be created based on this

def cifar_user_dataset(dataset, num_users, noniid_fraction):
    """
    Sample a 'fraction' of non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :param fraction:
    :return:
    """

    # initialization
    total_items = len(dataset)
    num_noniid_items = len(dataset) * noniid_fraction
    num_iid_items = total_items - num_noniid_items
    dict_users = list()
    for ii in range(num_users):
        dict_users.append(list())
    idxs = [i for i in range(len(dataset))]

    # IID
    # 밑에 dataset_iid 함수랑 똑같은 동작.
    if num_iid_items != 0:
        per_user_iid_items = int(num_iid_items / num_users) # 각 iid user가 가지는 데이터 개수
        for ii in range(num_users):
            tmp_set = set(np.random.choice(idxs, per_user_iid_items, replace=False))
            dict_users[ii] += tmp_set
            idxs = list(set(idxs) - tmp_set)

    # NON-IID
    if num_noniid_items != 0:

        num_shards = num_users  # each user has one shard
        per_shards_num_imgs = int(num_noniid_items / num_shards) # 각 non-iid user가 가지는 데이터 개수
        idx_shard = [i for i in range(num_shards)]
        labels = list()
        for ii in range(len(idxs)):
            labels.append(dataset[idxs[ii]][1]) # label들만 따로 labels 리스트에 저장
        print(labels)
        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        # for i in range(len(idxs_labels)):
        #     print('aaaaaaaaaaaaaaaaaaaaaaaaaaa')
        #     print(idxs_labels[i])
        idxs = idxs_labels[0, :] # iid index 빼고 non-iid index가 정렬됨.

        # divide and assign
        i = 0
        while idx_shard:
            print(idx_shard)
            rand_idx = np.random.choice(idx_shard, 1, replace=False) # random하게 shard(client) 하나 선택
            rand_idx[0] = idx_shard[0] # 첫번째 shard(client) 선택 (위에꺼 덮어씀. 왜??)
            # rand_idx.append(idx_shard[0])
            print(rand_idx)
            idx_shard = list(set(idx_shard) - set(rand_idx))
            dict_users[i].extend(idxs[int(rand_idx) * per_shards_num_imgs: (int(rand_idx) + 1) * per_shards_num_imgs])
            i = divmod(i + 1, num_users)[1]

    '''
    for ii in range(num_users):
        tmp = list()
        for jj in range(len(dict_users[ii])):
            tmp.append(dataset[dict_users[ii][jj]][1])
        tmp.sort()
        print(tmp)
    '''
    return dict_users

# dataset을 np.random.choice로 무작위로 num_users개의 client에 dict_users[i]로 나누어줌.
# 왜 np.ranoom.choice로 하면 iid가 보장되지?
def dataset_iid(dataset, num_users):
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def dirichlet_distribution_dict_users(targets, num_clients=10, alpha=0.1, min_size=0):
    """
    targets:        1D array-like of class labels (length = dataset size)
    num_clients:    number of clients
    alpha:          Dirichlet concentration parameter (small -> more heterogeneous)
    min_size:       ensure each client has at least this many samples (if >0, function will retry sampling)
    
    Returns:
        dict_users: list of lists, dict_users[i] is a list of dataset indices assigned to client i
    """

    targets = np.array(targets) # 전체 데이터셋의 label을 넣음.
    num_classes = int(targets.max()) + 1 # 전체 class 개수.
    class_idx = [np.where(targets==k)[0].tolist() for k in range(num_classes)] # 5000개씩 class 별로 나눔.
    
    # initialize list for each client
    dict_users = [[] for _ in range(num_clients)]
    
    # For each class, draw a Dirichlet distribution over clients, then use multinomial to get integer counts
    for k in range(num_classes):
        idx_k = class_idx[k] # 여기서 k는 class 번호를 뜻함. 즉, 0~9까지의 class 번호를 뜻함.
        n_k = len(idx_k)
        # logger.info(f"{k}번째 class의 데이터 개수: {n_k}")
        if n_k == 0:
            continue
        
        # draw proportions for this class
        p = np.random.dirichlet([alpha] * num_clients) # 각 client에 대한 비율을 랜덤하게 생성. # 현재 class에 대한 client 수만큼의 확률을 생성
        # logger.info(f"{k}번째 class의 비율 shape: {p.shape}")
        # convert to integer counts that sum exactly to n_k
        counts = np.random.multinomial(n_k, p)
        # logger.info(f"{k}번째 class의 비율을 랜덤하게 생성한 데이터 개수: {counts}")

        # shuffle indices of this class and allocate according to counts
        np.random.shuffle(idx_k)
        start = 0
        for client_id, count in enumerate(counts):
            if count > 0:
                dict_users[client_id].extend(idx_k[start:start + count])
                start += count
    
    # optional: ensure minimum size per client by retrying (simple approach)
    if min_size > 0:
        for _trial in range(10):  # 몇 번 재시도 허용
            sizes = [len(u) for u in dict_users]
            if min(sizes) >= min_size:
                break
            # 재샘플링 전체 (간단하게 다시 시도)
            return dirichlet_distribution_dict_users(targets, num_clients, alpha, min_size=min_size)
    
    return dict_users