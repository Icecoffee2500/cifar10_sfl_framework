import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)

from src.datasets.fl_dataset import DatasetSplit
from src.utils.utils import calculate_accuracy, prRed, prGreen, AverageMeter

#TODO: client는 cfg를 받지 않도록 하기.

# Client-side functions associated with Training and Testing
class FLClientBase(object):
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
        self.client_id = idx
        self.device = device
        self.dataset_length = len(dataset_split_dict_train)

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

        self.untrainable_params = {}

        self.epoch_loss = AverageMeter()
        self.epoch_acc = AverageMeter()
        self.batch_loss = AverageMeter()
        self.batch_acc = AverageMeter()

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
                self.optimizer.step()
                              
                self.batch_loss.update(loss.item())
                self.batch_acc.update(acc.item())
            
            prRed(f"Client{self.client_id} Train => Local Epoch: {local_epoch}  \tAcc: {acc.item():.3f} \tLoss: {loss.item():.4f}", logger=logger)

            self.epoch_loss.update(self.batch_loss.avg)
            self.epoch_acc.update(self.batch_acc.avg)

            params_list = [p.detach().clone() for p in self.model.state_dict().values()]
            res = params_list, self.dataset_length
        
        return res, self.epoch_loss.avg, self.epoch_acc.avg
    
    def evaluate(self, global_params: OrderedDict[str, torch.Tensor]):
        """
        global_model로 평가를 하되, 각자의 test dataset으로 평가를 함.
        """
        self.set_parameters(global_params)
        self.model.eval()
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
                output = self.model(images)
                
                # calculate loss
                loss = self.loss_func(output, labels)
                # calculate accuracy
                acc = calculate_accuracy(output, labels)
                                 
                self.batch_loss.update(loss.item())
                self.batch_acc.update(acc.item())
            
            # prGreen(f"Client{self.client_id} Test =>                     \tAcc: {acc.item():.4f} \tLoss: {loss.item():.3f}", logger=logger)
            prGreen(f"Client{self.client_id} Test =>\t\t\t\tAcc: {acc.item():.4f} \tLoss: {loss.item():.3f}", logger=logger)
            self.epoch_loss.update(self.batch_loss.avg)
            self.epoch_acc.update(self.batch_acc.avg)
        return self.epoch_loss.avg, self.epoch_acc.avg

    def set_parameters(self, model_params: OrderedDict):
        self.model.load_state_dict(model_params, strict=False)
        if self.client_id in self.untrainable_params.keys():
            self.model.load_state_dict(
                self.untrainable_params[self.client_id], strict=False
            )