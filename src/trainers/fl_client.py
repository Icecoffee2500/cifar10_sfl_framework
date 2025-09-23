import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from src.datasets.fl_dataset import DatasetSplit
from src.utils.utils import calculate_accuracy, prRed, prGreen, AverageMeter
import logging
logger = logging.getLogger(__name__)

# Client-side functions associated with Training and Testing
class FLClient(object):
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
        self.idx = idx
        self.device = device

        self.train_loader = DataLoader(
            dataset=DatasetSplit(dataset_train, dataset_split_dict_train),
            batch_size=cfg.train.batch_size,
            shuffle=cfg.train.shuffle
        )
        self.test_loader = DataLoader(
            dataset=DatasetSplit(dataset_test, dataset_split_dict_test),
            batch_size=cfg.test.batch_size,
            shuffle=cfg.test.shuffle
        )
        
        self.local_epochs = cfg.train.local_epochs
        self.loss_func = nn.CrossEntropyLoss()
        self.model = model
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.momentum,
            weight_decay=cfg.optimizer.weight_decay
        )

        self.epoch_loss = AverageMeter()
        self.epoch_acc = AverageMeter()
        self.batch_loss = AverageMeter()
        self.batch_acc = AverageMeter()

    def train(self):
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
                self.optimizer.step()
                              
                self.batch_loss.update(loss.item())
                self.batch_acc.update(acc.item())
            
            prRed(f"Client{self.idx} Train => Local Epoch: {local_epoch}  \tAcc: {acc.item():.3f} \tLoss: {loss.item():.4f}", logger=logger)

            self.epoch_loss.update(self.batch_loss.avg)
            self.epoch_acc.update(self.batch_acc.avg)
        
        return self.model.state_dict(), self.epoch_loss.avg, self.epoch_acc.avg
    
    def train_fedprox(self, global_model_weights_dict, mu):
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
                loss.backward() # weight.grad에 값이 저장됨.

                # FedProx일 경우.
                # local 모델의 named_parameters와 순회하면서
                if global_model_weights_dict is not None and mu is not None:
                    for name, local_param in self.model.named_parameters():
                        if name not in global_model_weights_dict:
                            # 이 이름은 global에 없으므로 건너뛰거나 warning
                            logger.warning(f"[no global param] {name}")
                            continue

                        global_param = global_model_weights_dict[name]
                        # 크기 체크
                        if local_param.shape != global_param.shape:
                            logger.warning(f"[mismatch] {name}: local {tuple(local_param.shape)} vs global {tuple(global_param.shape)}")
                            continue
                        
                        # FedProx 적용
                        if local_param.grad is None:
                            local_param.grad = torch.zeros_like(local_param)
                        with torch.no_grad():
                            delta = mu * (global_param.detach() - local_param.detach())
                            local_param.grad.add_(delta)

                self.optimizer.step()
                              
                self.batch_loss.update(loss.item())
                self.batch_acc.update(acc.item())
            
            prRed(f"Client{self.idx} Train => Local Epoch: {local_epoch}  \tAcc: {acc.item():.3f} \tLoss: {loss.item():.4f}", logger=logger)

            self.epoch_loss.update(self.batch_loss.avg)
            self.epoch_acc.update(self.batch_acc.avg)
        
        return self.model.state_dict(), self.epoch_loss.avg, self.epoch_acc.avg
    
    def evaluate(self, global_model):
        """
        global_model로 평가를 하되, 각자의 test dataset으로 평가를 함.
        """
        global_model.eval()
        self.epoch_acc.reset()
        self.epoch_loss.reset()
        self.batch_acc.reset()
        self.batch_loss.reset()

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                self.batch_loss.reset()
                self.batch_acc.reset()
                
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                output = global_model(images)
                
                # calculate loss
                loss = self.loss_func(output, labels)
                # calculate accuracy
                acc = calculate_accuracy(output, labels)
                                 
                self.batch_loss.update(loss.item())
                self.batch_acc.update(acc.item())
            
            # prGreen(f"Client{self.idx} Test =>                     \tAcc: {acc.item():.4f} \tLoss: {loss.item():.3f}", logger=logger)
            prGreen(f"Client{self.idx} Test =>\t\t\t\tAcc: {acc.item():.4f} \tLoss: {loss.item():.3f}", logger=logger)
            self.epoch_loss.update(self.batch_loss.avg)
            self.epoch_acc.update(self.batch_acc.avg)
        return self.epoch_loss.avg, self.epoch_acc.avg