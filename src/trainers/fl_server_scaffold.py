import torch
import numpy as np
from collections import OrderedDict
from typing import Generator

from src.trainers.fl_server_base import FLBaseServer
from src.trainers.fl_client_scaffold import ScaffoldClient
from src.utils.utils import prGreen, prRed, clone_parameters, metrics_log

class FLServerScaffold(FLBaseServer):
    def __init__(self, cfg, logger, wandb, device, clients: list[ScaffoldClient], global_params_dict: OrderedDict[str : torch.Tensor], global_parameters: Generator):
        super().__init__(cfg, logger, wandb, device, clients, global_params_dict)
        
        # self.global_parameters = global_parameters
        self.c_global = [
            torch.zeros_like(param).to(self.device)
            for param in global_parameters
        ]

        self.global_lr = 1.0
    
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
            res, loss_train, acc_train = client.train(global_params=client_local_params, c_global=self.c_global)

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

    @torch.no_grad()
    def aggregate(self, res_cache):
        """
        res_cache: list of (y_delta_list, c_delta_list)
        """
        y_delta_cache = list(zip(*res_cache))[0] # tuple of params_list
        c_delta_cache = list(zip(*res_cache))[1] # tuple of number_of_samples

        trainable_parameter = filter(
            lambda p: p.requires_grad, self.global_params_dict.values()
        )

        # update global model
        avg_weight = torch.tensor(
            [
                1 / len(self.clients)
                for _ in range(len(self.clients))
            ],
            device=self.device,
        )

        for param, y_del in zip(trainable_parameter, zip(*y_delta_cache)):
            x_del = torch.sum(avg_weight * torch.stack(y_del, dim=-1), dim=-1)
            # param.data += self.global_lr * x_del
            param.add_(self.global_lr * x_del)

        # update global control
        for c_g, c_del in zip(self.c_global, zip(*c_delta_cache)):
            c_del = torch.sum(avg_weight * torch.stack(c_del, dim=-1), dim=-1)
            # c_g.data += self.cfg.participation_rate * c_del
            c_g.add_(self.cfg.participation_rate * c_del)