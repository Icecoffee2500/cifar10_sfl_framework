import torch
from utils.utils import calculate_accuracy
from utils.utils import prRed, prGreen
import logging

logger = logging.getLogger(__name__)
# Server-side function associated with Training
class SLServer:
    def __init__(
        self, net_glob_server, criterion, device, num_users
    ):
        
        self.net_glob_server = net_glob_server
        self.criterion = criterion
        self.device = device
        self.num_users = num_users

        self.l_epoch_check = False
        self.fed_check = False
        self.count1 = 0
        self.count2 = 0
        self.acc_avg_all_user_train = 0
        self.loss_avg_all_user_train = 0
        self.idx_collect = []

        self.batch_loss_train = []
        self.batch_acc_train = []
        self.loss_train_collect = []
        self.loss_train_collect_user = []
        self.acc_train_collect = []
        self.acc_train_collect_user = []
        self.loss_test_collect = []
        self.loss_test_collect_user = []
        self.acc_test_collect = []
        self.batch_acc_test = []
        self.batch_loss_test = []
        self.acc_test_collect_user = []

    def train_server(self, fx_client, y, l_epoch_count, l_epoch, idx, len_batch, frac):
        self.net_glob_server.train()
        optimizer_server = torch.optim.SGD(self.net_glob_server.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        
        # train and update
        optimizer_server.zero_grad()
        
        fx_client = fx_client.to(self.device)
        y = y.to(self.device)
        
        #---------forward prop-------------
        fx_server = self.net_glob_server(fx_client)
        
        # calculate loss
        loss = self.criterion(fx_server, y)
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)
        
        #--------backward prop--------------
        loss.backward()
        dfx_client = fx_client.grad.clone().detach()
        optimizer_server.step()
        
        self.batch_loss_train.append(loss.item())
        self.batch_acc_train.append(acc.item())
        
        # server-side model net_glob_server is global so it is updated automatically in each pass to this function
            # count1: to track the completion of the local batch associated with one client
        self.count1 += 1
        if self.count1 == len_batch:
            acc_avg_train = sum(self.batch_acc_train)/len(self.batch_acc_train)           # it has accuracy for one batch
            loss_avg_train = sum(self.batch_loss_train)/len(self.batch_loss_train)
            
            self.batch_acc_train = []
            self.batch_loss_train = []
            self.count1 = 0
            
            prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train, loss_avg_train), logger=logger)
            
                    
            # If one local epoch is completed, after this a new client will come
            if l_epoch_count == l_epoch-1:
                
                self.l_epoch_check = True                # for evaluate_server function - to check local epoch has hitted 
                        
                # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
                # this is because we work on the last trained model and its accuracy (not earlier cases)
                
                #print("accuracy = ", acc_avg_train)
                acc_avg_train_all = acc_avg_train
                loss_avg_train_all = loss_avg_train
                            
                # accumulate accuracy and loss for each new user
                self.loss_train_collect_user.append(loss_avg_train_all)
                self.acc_train_collect_user.append(acc_avg_train_all)
                
                # collect the id of each new user                        
                if idx not in self.idx_collect:
                    self.idx_collect.append(idx) 
                    #print(idx_collect)
            
            # This is to check if all users are served for one round --------------------
            if len(self.idx_collect) == self.num_users * frac:
                self.fed_check = True # for evaluate_server function  - to check fed check has hitted
                # all users served for one round ------------------------- output print and update is done in evaluate_server()
                # for nicer display 
                            
                self.idx_collect = []
                
                self.acc_avg_all_user_train = sum(self.acc_train_collect_user)/len(self.acc_train_collect_user)
                self.loss_avg_all_user_train = sum(self.loss_train_collect_user)/len(self.loss_train_collect_user)
                
                self.loss_train_collect.append(self.loss_avg_all_user_train)
                self.acc_train_collect.append(self.acc_avg_all_user_train)
                
                self.acc_train_collect_user = []
                self.loss_train_collect_user = []
                
        # send gradients to the client               
        return dfx_client

    # Server-side functions associated with Testing
    def evaluate_server(self, fx_client, y, idx, len_batch, ell):
        self.net_glob_server.eval()
    
        with torch.no_grad():
            fx_client = fx_client.to(self.device)
            y = y.to(self.device) 
            #---------forward prop-------------
            fx_server = self.net_glob_server(fx_client)
            
            # calculate loss
            loss = self.criterion(fx_server, y)
            # calculate accuracy
            acc = calculate_accuracy(fx_server, y)
            
            
            self.batch_loss_test.append(loss.item())
            self.batch_acc_test.append(acc.item())
            
                
            self.count2 += 1
            if self.count2 == len_batch:
                acc_avg_test = sum(self.batch_acc_test)/len(self.batch_acc_test)
                loss_avg_test = sum(self.batch_loss_test)/len(self.batch_loss_test)
                
                self.batch_acc_test = []
                self.batch_loss_test = []
                self.count2 = 0
                
                prGreen('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test, loss_avg_test), logger=logger)
                
                # if a local epoch is completed   
                if self.l_epoch_check:
                    self.l_epoch_check = False
                    
                    # Store the last accuracy and loss
                    acc_avg_test_all = acc_avg_test
                    loss_avg_test_all = loss_avg_test
                            
                    self.loss_test_collect_user.append(loss_avg_test_all)
                    self.acc_test_collect_user.append(acc_avg_test_all)
                    
                # if all users are served for one round ----------                    
                if self.fed_check:
                    self.fed_check = False
                                    
                    acc_avg_all_user = sum(self.acc_test_collect_user)/len(self.acc_test_collect_user)
                    loss_avg_all_user = sum(self.loss_test_collect_user)/len(self.loss_test_collect_user)
                
                    self.loss_test_collect.append(loss_avg_all_user)
                    self.acc_test_collect.append(acc_avg_all_user)
                    self.acc_test_collect_user = []
                    self.loss_test_collect_user= []
                                
                    logger.info("====================== SERVER V1==========================")
                    logger.info(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, self.acc_avg_all_user_train, self.loss_avg_all_user_train))
                    logger.info(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user, loss_avg_all_user))
                    logger.info("==========================================================")

                    return {
                        "train/acc_avg": self.acc_avg_all_user_train,
                        "train/loss_avg": self.loss_avg_all_user_train,
                        "test/acc_avg": acc_avg_all_user,
                        "test/loss_avg": loss_avg_all_user
                    }
            
        return None