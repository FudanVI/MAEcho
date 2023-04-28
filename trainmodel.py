import sys, time
import numpy as np
import torch


torch.manual_seed(0)
torch.cuda.manual_seed(0) 
np.random.seed(0) 



########################################################################################################################

class Appr(object):
    def __init__(self, model,args=None):
        self.model = model
        self.nepochs = args.num_epochs
        self.sbatch = args.batch_size
        self.lr = args.learning_rate
        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.gpu = args.gpu_id


        return
    
    def update_lr(self,optimizer):    
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/10.
            
            
    def _get_optimizer(self, lr=None):
        lr = self.lr
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,momentum = 0.5)
        return optimizer

    def train(self, train_loader):

        lr = self.lr
        self.optimizer = self._get_optimizer(lr)
        nepochs = self.nepochs
        # Loop epochs
        try:
            for e in range(nepochs):
                self.train_epoch(train_loader, cur_epoch=e, nepoch=nepochs)
        except KeyboardInterrupt:
            print()


    def train_epoch(self,train_loader, cur_epoch=0, nepoch=0):
        self.model.train()
        for i, (images, labels) in enumerate(train_loader):

            images = images.cuda(self.gpu)
            targets = labels.cuda(self.gpu)
            output,_= self.model.forward(images)
            loss = self.ce(output, targets)
            self.optimizer.zero_grad()
            loss.backward()  
            self.optimizer.step()
        return 
    

        