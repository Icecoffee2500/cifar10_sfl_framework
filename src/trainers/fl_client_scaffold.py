from src.trainers.fl_client_base import FLClientBase
import torch
from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)

from src.utils.utils import calculate_accuracy, prRed

torch.autograd.set_detect_anomaly(True)


def detect_nan_inf(prefix, tensors):
    for name, t in tensors.items():
        if not torch.isfinite(t).all():
            print(f"[{prefix}] NON-FINITE: {name}: min {t.min().item() if t.numel() else 'NA'}, max {t.max().item() if t.numel() else 'NA'} device {t.device}")
            return True
    return False

class ScaffoldClient(FLClientBase):
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

        # self.c_local: dict[list[torch.Tensor]] = {}
        self.c_local: list[torch.Tensor] = []
        self.c_diff = []

    def train(self, global_params: OrderedDict[str, torch.Tensor], c_global):
        self.set_parameters(global_params)
        self.model.train()
        
        self.epoch_loss.reset()
        self.epoch_acc.reset()

        # SCAFFOLD 적용 ------------------------------
        if self.c_local == []:
            self.c_diff = c_global
        else:
            # c_diff는 aggregate된 c_global과 아직 학습하지 않은 c_local의 차이. (c_local은 이전 round의 c_plus)
            self.c_diff = []
            for c_l, c_g in zip(self.c_local, c_global):
                self.c_diff.append(c_g - c_l)
        # -------------------------------------------

        update_count = 0
        # for local_epoch in range(self.local_epochs):
        for local_epoch in range(1):
            self.batch_loss.reset()
            self.batch_acc.reset()
            
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                update_count += 1
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

                # SCAFFOLD 적용 ------------------------------
                with torch.no_grad():
                    # for param, c_d in zip(self.model.parameters(), self.c_diff):
                    for param, c_d in zip(self.model.state_dict().values(), self.c_diff):
                        if not param.requires_grad:
                            continue
                        param.grad.add_(c_d.detach())
                # -------------------------------------------
                self.optimizer.step()
                              
                self.batch_loss.update(loss.item())
                self.batch_acc.update(acc.item())
            
            prRed(f"Client{self.client_id} Train => Local Epoch: {local_epoch}  \tAcc: {acc.item():.3f} \tLoss: {loss.item():.4f}", logger=logger)

            self.epoch_loss.update(self.batch_loss.avg)
            self.epoch_acc.update(self.batch_acc.avg)
        
        # SCAFFOLD 적용 ------------------------------
        with torch.no_grad():
            trainable_parameters = filter(
                lambda p: p.requires_grad, global_params.values() # global_params 중에서 requires_grad가 true인 파라미터만 필터링.
            )

            if self.c_local == []: # c_local을 초기화.
                # self.c_local = [torch.zeros_like(param, device=self.device) for param in self.model.parameters()]
                self.c_local = [torch.zeros_like(param, device=self.device) for param in self.model.state_dict().values()]

            y_delta = []
            c_plus = []
            c_delta = []

            # compute y_delta (difference of model before and after training)

            # for param_l, param_g in zip(self.model.parameters(), trainable_parameters):
            #     y_delta.append(param_l - param_g)
            for param_l, param_g in zip(self.model.state_dict().values(), global_params.values()):
                y_delta.append(param_l - param_g)
            
            # compute c_plus # Option II version
            # coef = 1 / (self.local_epochs * self.local_lr)
            coef = 1 / (update_count * self.local_lr)
            
            for c_l, c_g, diff in zip(self.c_local, c_global, y_delta):
                c_plus.append(c_l - c_g - coef * diff)

            # compute c_delta
            for c_p, c_l in zip(c_plus, self.c_local):
                c_delta.append(c_p - c_l)

            self.c_local = c_plus

            # for name, param in self.model.state_dict(keep_vars=True).items():
            #     if not param.requires_grad:
            #         self.untrainable_params[name] = param.clone()
        
        # res = (y_delta, c_delta)
        res = (y_delta, self.dataset_length, c_delta)
        # -------------------------------------------
        return res, self.epoch_loss.avg, self.epoch_acc.avg