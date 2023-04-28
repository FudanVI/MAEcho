
import torch
import torch.nn as nn
import torch.nn.functional as F
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) 


class cnnNet(torch.nn.Module):

    def __init__(self):
        super(cnnNet, self).__init__()
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(2)
        self.drop1 = torch.nn.Dropout(0.2)
        self.padding = torch.nn.ReplicationPad2d(1)
        
        self.c1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.c2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1,  padding=1, bias=False)
        self.c3 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)

        self.fc1 = torch.nn.Linear(256 * 4 * 4, 1000, bias=False)
        self.fc2 = torch.nn.Linear(1000, 500, bias=False)
        self.fc4 = torch.nn.Linear(500, 10,  bias=False)

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc4.weight)

        return
    

    def forward(self, x,):
        h_list = []
        h_list.append(torch.mean(x, 0, True))
        con1 = self.drop1(self.relu(self.c1(x)))
        con1_p = self.maxpool(con1)

        h_list.append(torch.mean(con1_p, 0, True))
        con2 = self.drop1(self.relu(self.c2(con1_p)))
        con2_p = self.maxpool(con2)

        h_list.append(torch.mean(con2_p, 0, True))
        con3 = self.drop1(self.relu(self.c3(con2_p)))
        con3_p = self.maxpool(con3)

        h = con3_p.view(x.size(0), -1)
        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc1(h))

        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc2(h))

        h_list.append(torch.mean(h, 0, True))
        y = self.fc4(h)
        return y, h_list
    

    
class mnistnet(nn.Module):
    def __init__(self):
        super(mnistnet, self).__init__()
        self.fc1 = nn.Linear(784, 400,bias=False)
        self.fc2 = nn.Linear(400, 200,bias=False)
        self.fc3 = nn.Linear(200, 100,bias=False)
        self.fc4 = nn.Linear(100, 10,bias=False)

    def forward(self, x):

        hlist=[]

        x = x.view(-1, 784)
        hlist.append(torch.mean(x, 0, True))
        x = F.relu(self.fc1(x))
        hlist.append(torch.mean(x, 0, True))
        x = F.relu(self.fc2(x))
        hlist.append(torch.mean(x, 0, True))
        x = F.relu(self.fc3(x))
        hlist.append(torch.mean(x, 0, True))
        x = self.fc4(x)

        return x,hlist
    
