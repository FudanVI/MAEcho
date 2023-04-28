import sys, time
import numpy as np
import torch
import copy
import random
import torch.nn.functional as F
from tqdm import tqdm
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
torch.cuda.manual_seed(0) 
np.random.seed(0) 
random.seed(0)


########################################################################################################################



def createp(model, train_loader,args=None):
    model.eval()
    PP = {}
    for i,layer in model.named_parameters():
        if len(layer.shape)==4:
            PP[i]=torch.eye(layer.shape[1]*layer.shape[2]*layer.shape[3], requires_grad=False).cuda(args.gpu_id)
        elif len(layer.shape)==2:
            PP[i]=torch.eye(layer.shape[1], requires_grad=False).cuda(args.gpu_id)
    r_len=args.batch_size*len(list(enumerate(train_loader)))
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda(args.gpu_id)
        targets = labels.cuda(args.gpu_id)
        output, h_list= model.forward(images)
        lamda = 0.8
        alpha_array = [1.0 * 0.00001 ** lamda, 1.0 * 0.0001 ** lamda, 1.0 * 0.01 ** lamda, 1.0 * 0.1 ** lamda]
        def pro_weight1(p, x, w, alpha=1.0, cnn=True, stride=1,so=0.5):
            if cnn:
                _, _, H, W = x.shape
                F, _, HH, WW = w.shape
                S = stride  
                Ho = int(1 + (H - HH) / S)
                Wo = int(1 + (W - WW) / S)
                for i in range(Ho):
                    for j in range(Wo):
                        r = x[:, :, i * S: i * S + HH, j * S: j * S + WW].contiguous().view(1, -1)
                        k = torch.mm(p, torch.t(r))
                        p.sub_(so*torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
            else:
                r = x
                k = torch.mm(p, torch.t(r))
                p.sub_(so*torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
                
        with torch.no_grad():
            if args.model_type=='mlpnet':
                index = 0
                for n, w in model.named_parameters():
                    if len(w.shape)!=2: continue
                    else:
                        pro_weight1(PP[n],  h_list[index], w, alpha=alpha_array[2], cnn=False,so=args.C)
                        index += 1    
            elif args.model_type=='cnnnet':
                index = 0
                for n, w in model.named_parameters():
                    if len(w.shape)==4: 
                        pro_weight1(PP[n],  h_list[index], w, alpha=alpha_array[0], stride=2,so=args.C)
                        index += 1 
                    elif len(w.shape)==2:
                        pro_weight1(PP[n],  h_list[index], w, alpha=alpha_array[2], cnn=False,so=args.C)
                        index += 1   
    return  PP


def valid(model,test,args=None):
    model.eval()
    correct = 0
    total = 0
    for images, labels in test:
        images = images.cuda(args.gpu_id)
        labels = labels.cuda(args.gpu_id)
        outputs,_ = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100 * correct / total


                  
def record_net_data_stats(y_train, net_dataidx_map):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    return net_cls_counts

def partition(alphaa, n_netss, y_train,):
    min_size = 0
    n_nets = n_netss
    N = y_train.shape[0]
    net_dataidx_map = {}
    alpha = alphaa
    K=10
    while min_size < 10:
        idx_batch = [[] for _ in range(n_nets)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return net_dataidx_map,traindata_cls_counts


def noniid_ensemble(models,test,splits,args=None):    
    if args.split:
        all_classes = list(range(10)) 
        t_cla = np.array([0]*10)
        for i in splits:
            t_cla[i]+=1

        key ='fc4.weight'
        temps  = copy.deepcopy(models)
        for i in range(0, len(temps)):
            temps[i].state_dict()[key][list(set(all_classes).difference(set(splits[i])))] *= 0


        correct = 0
        total = 0
        for images, labels in test:
            images = images.cuda(args.gpu_id)
            labels = labels.cuda(args.gpu_id)
            outputs,_ = temps[0](images)
            for i,model in enumerate(temps[1:]):
                out,_ = model(images)
                outputs += out
            for i in range(outputs.shape[1]):
                if t_cla[i] ==0:continue
                else :
                    outputs[:,i]/=t_cla[i]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return 100 * correct / total
    else:
        was_training = [False]*len(models)
        for i, model in enumerate(models):
            if model.training:
                was_training[i] = True
                model.eval()

        correct = 0
        total = 0
        for images, labels in test:
            images = images.cuda(args.gpu_id)
            labels = labels.cuda(args.gpu_id)
            outputs,_ = models[0](images)
            for i,model in enumerate(models[1:]):
                out,_ = model(images)
                outputs += out
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        for i, model in enumerate(models):
            if was_training[i]:
                model.train()

        return 100 * correct / total
        



def noniid_avgmodel_o(w,splits,split=True):
    if split:
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.state_dict().keys():
            if key =='fc4.weight':continue
            for i in range(1, len(w)):
                w_avg.state_dict()[key] += (w[i].state_dict()[key])
            w_avg.state_dict()[key] /= len(w)
        key ='fc4.weight'
        all_classes = list(range(10)) 
        t_cla = np.array([0]*10)
        for i in splits:
            t_cla[i]+=1

        temp  = []
        for i in range(0, len(w)):
            temp.append(copy.deepcopy(w[i].state_dict()[key]))
            temp[-1][list(set(all_classes).difference(set(splits[i])))] *= 0
        w_avg.state_dict()[key] += (temp[0]-w[0].state_dict()[key])
        for i in range(1, len(w)):
            w_avg.state_dict()[key] += (temp[i])
        for i in range(w_avg.state_dict()[key].shape[0]):
            if t_cla[i]==0:continue
            else:
                w_avg.state_dict()[key][i] /= t_cla[i]
        return w_avg
    else:
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.state_dict().keys():
            for i in range(1, len(w)):
                w_avg.state_dict()[key] += (w[i].state_dict()[key])
            w_avg.state_dict()[key] /= len(w)
        return w_avg

def noniid_avgmodel(w,splits,split=True):
    if split:
        all_classes = list(range(10)) 
        t_cla = np.array([0]*10)
        for i in splits:
            t_cla[i]+=1

        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            if key == 'fc4.weight':
                temp  = []
                for i in range(0, len(w)):
                    temp.append(copy.deepcopy(w[i][key]))
                    temp[-1][list(set(all_classes).difference(set(splits[i])))] *= 0
                w_avg[key] += (temp[0]-w[0][key])
                for i in range(1, len(w)):
                    w_avg[key] += temp[i]
                for i in range(w_avg[key].shape[0]):
                    if t_cla[i]==0:continue
                    else:
                        w_avg[key][i] /= t_cla[i]
            else:
                for i in range(1, len(w)):
                    w_avg[key] += w[i][key]
                w_avg[key] /= len(w)
        return w_avg
    else:
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] /= len(w)
        return w_avg


import cvxopt
import copy
import cvxopt.solvers
cvxopt.solvers.options['show_progress'] = False
class SVM():
    def __init__(self,C=0.5, kernel=None):
        self.kernel = kernel                          
        self.C = float(C)        
        
    def fit(self, w_now, ws,ps):
        N = len(ws)
#         M = w_now.shape[1]
        AA = torch.Tensor(N,N)
        for i in range(N):
            for j in range(N):
                k1 = torch.mm(w_now-ws[i],ps[i])
                if j==i: 
                    k2 = (w_now-ws[j]).T
                else:
                    k2 = torch.mm(ps[j],(w_now-ws[j]).T)
                AA[i,j]=torch.trace(torch.mm(k1,k2))
                
        P = AA
        
        P = cvxopt.matrix(np.array(P).astype(np.double))       
        q = cvxopt.matrix(np.zeros((N,1)))  
        A = cvxopt.matrix(np.ones((1,N)))       
        b = cvxopt.matrix(1.0)                     
        
        tmp1 = np.diag(np.ones(N) * -1)
        tmp2 = np.identity(N)      
        G = cvxopt.matrix(np.vstack((tmp1, tmp2))) 
        tmp1 = np.zeros(N)
        tmp2 = np.ones(N) * self.C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)['x'] 
        
        return solution
    
def noniid_aggr(w_list,pps,testdata,splits=None,args=None):
    """
    w=[w0,w1...]
    p=[p0,p1,p2...]
    """
    modelout = copy.deepcopy(w_list[0])
    temp1 = copy.deepcopy(w_list[0])
    temp2 = copy.deepcopy(w_list[0])
    temp3 = copy.deepcopy(w_list[0])
    out_acc = []
    N = len(w_list)
    w = [copy.deepcopy(w_list[i].state_dict()) for i in range(N)]
    wout = noniid_avgmodel(w,splits)
    alpha = [1/N]*N
    for numm in tqdm(range(args.maxt_times)): 
        if args.maxt_times > 100: printstep=10
        else: printstep=1
        if numm%printstep==0:
            if args.test:
                acc = []
                for tt in list(range(args.n_nets)):
                    modelout.load_state_dict(w[tt])
                    acc.append(valid(modelout,testdata,args))
                modelout.load_state_dict(noniid_avgmodel(w,splits,args.split))
                acc.append(valid(modelout,testdata,args))
                out_acc.append(acc)  
                
        for li, layer in enumerate(w[0]):
            F = w[0][layer].shape[0]
            if layer==list(w[0].keys())[-1]:continue
#             alpha = np.ravel(svm.fit(wout[layer].view(F, -1),[w[n][layer].view(F, -1) for n in range(N)],[pps[m][layer]-torch.eye(pps[m][layer].shape[0], requires_grad=False).cuda(args.gpu_id) for m in range(N)] ))

            D = args.lambdastep*alpha[0]*torch.mm((wout[layer]-w[0][layer]).view(F, -1), torch.t(pps[0][layer]-torch.eye(pps[0][layer].shape[0], requires_grad=False).cuda(args.gpu_id)))
            for mi in range(1,len(w)):
                dd = torch.mm((wout[layer]-w[mi][layer]).view(F, -1), torch.t(pps[mi][layer]-torch.eye(pps[mi][layer].shape[0], requires_grad=False).cuda(args.gpu_id)))
                D += args.lambdastep*alpha[mi]*dd
            wout[layer] = wout[layer] + D.view_as(w[0][layer])

            for m_num in range(len(w)): 
                if args.norm: 
                    dd = torch.mm((wout[layer]-w[m_num][layer]).view(F, -1), torch.t(pps[m_num][layer]))
                    w[m_num][layer] = w[m_num][layer]+args.lambdastep*(dd/torch.norm(dd,dim=1).unsqueeze(1)).view_as(w[m_num][layer])
                else:

                    w[m_num][layer] = w[m_num][layer]+args.lambdastep*torch.mm((wout[layer]-w[m_num][layer]).view(F, -1), torch.t(pps[m_num][layer])).view_as(w[m_num][layer])

                    
    if args.test:
        acc = []
        for tt in list(range(args.n_nets)):
            modelout.load_state_dict(w[tt])
            acc.append(valid(modelout,testdata,args))
        modelout.load_state_dict(noniid_avgmodel(w,splits,args.split))
        acc.append(valid(modelout,testdata,args))
        print('global model acc',acc[-1])
        out_acc.append(acc)
        
        return np.array(out_acc) 
    else:
        modelout.load_state_dict(noniid_avgmodel(w,splits,args.split))
        final_acc = valid(modelout,testdata,args)
        print('global model acc',final_acc)
        
        return final_acc
        


