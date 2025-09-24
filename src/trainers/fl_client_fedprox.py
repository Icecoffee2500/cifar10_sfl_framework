from src.trainers.fl_client_base import FLClientBase
import torch
from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)

from src.utils.utils import calculate_accuracy, prRed

class FedProxClient(FLClientBase):
    def __init__(
        self,
        cfg,
        idx,
        device,
        model,
        dataset_train=None,
        dataset_test=None,
        dataset_split_dict_train=None,
        dataset_split_dict_test=None
    ):
        super().__init__(cfg, idx, device, model, dataset_train, dataset_test, dataset_split_dict_train, dataset_split_dict_test)

        self.trainable_global_params: list[torch.Tensor] = None
        self.mu = cfg.train.hyperparameter.mu

    def train(self, global_params: OrderedDict[str, torch.Tensor]):
        self.set_parameters(global_params)
        self.model.train()
        
        self.epoch_loss.reset()
        self.epoch_acc.reset()

        for local_epoch in range(self.local_epochs):
            self.batch_loss.reset()
            self.batch_acc.reset()
            
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                #---------forward prop-------------
                output = self.model(images)
                
                # calculate loss
                loss = self.loss_func(output, labels)
                
                # calculate accuracy
                acc = calculate_accuracy(output, labels)
                
                #--------backward prop--------------
                loss.backward()

                # FedProx 적용
                with torch.no_grad():
                    for w, w_g in zip(self.model.parameters(), self.trainable_global_params):
                        # w.grad.data += self.mu * (w_g.data - w.data)
                        if w.grad is None:
                            w.grad = torch.zeros_like(w)
                        
                        w.grad.add_(self.mu * (w_g - w))

                self.optimizer.step()
                              
                self.batch_loss.update(loss.item())
                self.batch_acc.update(acc.item())
            
            prRed(f"Client{self.client_id} Train => Local Epoch: {local_epoch}  \tAcc: {acc.item():.3f} \tLoss: {loss.item():.4f}", logger=logger)

            self.epoch_loss.update(self.batch_loss.avg)
            self.epoch_acc.update(self.batch_acc.avg)

            params_list = [p.detach().clone() for p in self.model.state_dict().values()]
            res = params_list, self.dataset_length
        
        return res, self.epoch_loss.avg, self.epoch_acc.avg

    def set_parameters(self, model_params: OrderedDict[str, torch.Tensor]):
        super().set_parameters(model_params)

        self.trainable_global_params = list(
            filter(lambda p: p.requires_grad, model_params.values())
        )