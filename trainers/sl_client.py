import torch
from torch.utils.data import DataLoader
from datasets.fl_dataset import DatasetSplit
from trainers.sl_server import SLServer
from torch import nn
import logging

logger = logging.getLogger(__name__)

class Client:
    def __init__(self, net_client_model, server, idx, lr, device, frac, total_num_users, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 5
        #self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size = 128, shuffle = True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size = 100, shuffle = True)
        self.frac = frac
        self.total_num_users = total_num_users
        self.server = server

    def train(self, net):
        net.train()
        optimizer_client = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        
        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                #---------forward prop-------------
                fx = net(images)
                client_fx = fx.clone().detach().requires_grad_(True)
                
                # Sending activations to server and receiving gradients from server
                dfx = self.server.train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch, self.frac)
                
                #--------backward prop -------------
                fx.backward(dfx)
                optimizer_client.step()
                            
            
            #prRed('Client{} Train => Epoch: {}'.format(self.idx, ell), logger=logger)
           
        return net.state_dict() 
    
    def evaluate(self, net, ell):
        net.eval()
           
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                fx = net(images)
                
                # Sending activations to server 
                wdb_log_dict = self.server.evaluate_server(fx, labels, self.idx, len_batch, ell)
            
            #prRed('Client{} Test => Epoch: {}'.format(self.idx, ell), logger=logger)
            
        return wdb_log_dict