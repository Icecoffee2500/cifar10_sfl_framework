import torch
import numpy as np
from collections import OrderedDict

from src.trainers.fl_client_base import FLClientBase
from src.utils.utils import prGreen, prRed, clone_parameters, metrics_log

class FLServerBase:
    def __init__(self, cfg, logger, wandb, device, clients: list[FLClientBase], global_params_dict: OrderedDict[str : torch.Tensor]):
        self.cfg = cfg
        self.logger = logger
        self.wandb = wandb
        self.device = device
        self.clients = clients
        self.global_params_dict = global_params_dict
        
        self.global_loss_train_collect = []
        self.global_acc_train_collect = []
        self.global_loss_test_collect = []
        self.global_acc_test_collect = []
    
    def train(self):
        for global_epoch in range(self.cfg.global_epochs):
            self.train_one_epoch(global_epoch)
            if global_epoch % self.cfg.save_interval == 0:
                prGreen(f"[Round {global_epoch}] best test acc: {max(self.global_acc_test_collect)}", logger=self.logger)
        
        prRed("Training and Evaluation completed!", logger=self.logger)
    
    @metrics_log
    def train_one_epoch(self, client_train_losses, client_train_accuracies, client_test_losses, client_test_accuracies):
        res_cache = []

        # 참여하는 client 선택 # 매 round마다 참여하는 client가 달라짐.
        num_users_participated = max(int(self.cfg.participation_rate * self.cfg.num_users), 1)
        participated_clients_idx = np.random.choice(range(self.cfg.num_users), num_users_participated, replace = False)

        # Train/Evaluate simulation
        for idx in participated_clients_idx: # each client
            # client 선택
            client = self.clients[idx]
            # Train ------------------
            client_local_params = clone_parameters(self.global_params_dict)
            res, loss_train, acc_train = client.train(global_params=client_local_params)

            res_cache.append(res) # client weight 저장.

            client_train_losses.update(loss_train)
            client_train_accuracies.update(acc_train)
            
            # Evaluate -------------------
            # loss_test, acc_test = client.evaluate(global_model=self.model_global)
            loss_test, acc_test = client.evaluate(self.global_params_dict)

            client_test_losses.update(loss_test)
            client_test_accuracies.update(acc_test)

        # Aggregating ==========================================================
        prGreen("------ Aggregating at Server -------", logger=self.logger)
        self.aggregate(res_cache)
        

    def test(self):
        pass

    @torch.no_grad()
    def aggregate(self, res_cache):
        """
        res_cache: list of (params_list, number_of_samples)
        """
        updated_params_cache = list(zip(*res_cache))[0] # tuple of params_list
        weights_cache = list(zip(*res_cache))[1] # tuple of number_of_samples
        weight_sum = sum(weights_cache) # total number of samples
        weights = torch.tensor(weights_cache, device=self.device) / weight_sum # 각 client에 곱해줄 가중치 (ratio of number of samples)

        aggregated_params = []

        for params in zip(*updated_params_cache): # zip(*updated_params_cache)은 각 개별 parameter별로 client들의 parameter를 묶는다.
            # 즉, 여기서 params는 모든 client들의 같은 파라미터들이다.
            aggregated_params.append(
                torch.sum(weights * torch.stack(params, dim=-1), dim=-1) # 결과의 shape은 원래 parameter의 shape과 같다.
            )

        self.global_params_dict = OrderedDict(
            zip(self.global_params_dict.keys(), aggregated_params)
        )