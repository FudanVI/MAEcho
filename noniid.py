import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from Subdata import  MNIST_truncated,CIFAR10_truncated
import torch.nn.functional as F
import numpy
import copy
import random
import trainmodel as approach
import modelset as network
from utils import partition,createp,valid,noniid_avgmodel_o,noniid_aggr,noniid_ensemble
import numpy as np
from do_ot import doot
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default=0.01,type=float,)
    parser.add_argument('--model_type', default='mlpnet', help='mlpnet for mnist and cnnnet for cifar')
    parser.add_argument('--data', default='mnist',help='mnist or cifar')
    parser.add_argument('--n_nets', default=5, type=int, help='model nums')
    parser.add_argument('--diff_init', default=True, help='True means different init')
    parser.add_argument('--maxt_times', default=50,type=int, help='iterate times')
    parser.add_argument('--C', default=0.5,type=float,)
    parser.add_argument('--norm', default=False)
    parser.add_argument('--test', default=True, help='If True, record all accuracy rates of each iteration. If just want to see accuracy of aggregation, set False.')
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU id to use')
    parser.add_argument('--seed', default=1,type=int,)
    parser.add_argument('--lambdastep', default=1.6,type=float, help='step_size')
    parser.add_argument('--split',  default=True, help='whether to discard the untrained parameters of the last layer')
    parser.add_argument('--repeat',  default=1, type=int,help='repeat times')
    parser.add_argument('--batch_size', default=64,type=int,)
    parser.add_argument('--learning_rate', default=0.01,type=float,)
    parser.add_argument('--num_epochs', default=10,type=int,)
    parser.add_argument('--alter', default=True, help='alternate or parallel')
    parser.add_argument('--logdir', default='logfinal')
    parser.add_argument('--expe', default='agg1',help='name of experiment')
    return parser


parser = get_parser()
args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) 
np.random.seed(seed) 
random.seed(seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

# dataset
if args.data == 'cifar':
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    train_data = torchvision.datasets.CIFAR10(root='./data/',train=True, download=True)
    test_data = torchvision.datasets.CIFAR10(root='./data/',train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
elif args.data  == 'mnist':
    mean = (0.1307,)
    std = (0.3081,)
    train_data = torchvision.datasets.MNIST(root='./data/',train=True,download=True)
    test_data =torchvision.datasets.MNIST(root='./data/',                              train=False,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

ytrain_label = np.array([train_data[i][1] for i in range(len(train_data))])

for r in range(args.repeat):
    models = []
    traindatas = []
    net_dataidx_map,traindata_cls_counts =partition(args.alpha, args.n_nets ,ytrain_label)
    
    splits=[[] for i in range(args.n_nets)]
    for mindex,classes in enumerate(splits):
            for i in traindata_cls_counts[mindex]:
                if traindata_cls_counts[mindex][i]>1:classes.append(i) #if images number <=1, then ignore the parameters of this class in the last layer when ensemble.
    
    #data partition
    for i in range(args.n_nets):
        if args.data=='mnist':
            traindata = MNIST_truncated(train_data,net_dataidx_map[i],transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]),)
        elif args.data=='cifar':
            traindata = CIFAR10_truncated(train_data,net_dataidx_map[i],transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]),)
        traindatas.append(torch.utils.data.DataLoader(dataset=traindata,batch_size=args.batch_size, shuffle=True))
    testdata = torch.utils.data.DataLoader(dataset=test_data,batch_size=args.batch_size, shuffle=False)


    if args.diff_init:
        if args.model_type=='mlpnet':
            for mnum in range(args.n_nets):
                models.append(network.mnistnet().cuda(args.gpu_id))
        elif args.model_type=='cnnnet':
            for mnum in range(args.n_nets):
                models.append(network.cnnNet().cuda(args.gpu_id))
    else:
        if args.model_type=='mlpnet':
            models.append(network.mnistnet().cuda(args.gpu_id))
        elif args.model_type=='cnnnet':
            models.append(network.cnnNet().cuda(args.gpu_id))
        for mnum in range(1,args.n_nets):
            models.append(copy.deepcopy(models[0]))
     
    acc = []
    print('Training...')
    for i,model in enumerate(models):
        appr = approach.Appr(model,args = args)
        appr.train(traindatas[i])
        acc.append(valid(model,testdata,args))
    print('Training is OK')
    print('all model acc in complete testset:', acc)
    ensemble_acc = noniid_ensemble(models,testdata,splits,args)
    
    modelots = []
    if args.n_nets>2:
        modelots = [models[0]]
        for model in models[1:]:
            modelots.append(doot([model,models[0]],args.model_type,args))
    else:
        modelots.append(doot([models[0],models[1]],args.model_type,args))
        modelots.append(models[1])
    

    ot_acc = valid(noniid_avgmodel_o(modelots,splits,args.split),testdata,args)
    fedavg_acc = valid(noniid_avgmodel_o(models,splits,args.split),testdata,args)
    
    print('fedavg_acc:',fedavg_acc)
    print('ot_acc:',ot_acc)
    print('ensemble_acc',ensemble_acc)
    
    print('prepare P')
    ppots = []
    pps = []
    for i,model in enumerate(modelots):
        ppots.append(createp(model,traindatas[i],args = args))
    for i,model in enumerate(models):
        pps.append(createp(model,traindatas[i],args = args))
        
    print('start aggregation ours+OT !')
    ours_ot = noniid_aggr(modelots,ppots,testdata,splits=splits,args = args)
    print('done!')
    print('start aggregation ours !')
    ours  = noniid_aggr(models,pps,testdata,splits=splits,args = args)
    print('done!')
    log = {
        'ours_ot':{
            'acc':ours_ot,
        },
        'ours':{
            'acc':ours,

        },
        'ensemble_acc':ensemble_acc,
        'ot_acc':ot_acc,
        'fedavg_acc':fedavg_acc,
        'all_model_acc':np.array(acc),
        'partition':traindata_cls_counts,
        }
    
    path='./'+args.logdir+'/'+str(args.diff_init)+'_'+args.expe+'_logdata_'+str(r)+'_'+str(args.n_nets)+'_'+args.data
    torch.save(log,path)
    
if args.repeat>1: 
    path='./'+args.logdir+'/'+str(args.diff_init)+'_'+args.expe+'_logdata_0_'+str(args.n_nets)+'_'+args.data
    logall = torch.load(path)
    for i in range(1,args.repeat):
        path= './'+args.logdir+'/'+str(args.diff_init)+'_'+args.expe+'_logdata_'+str(i)+'_'+str(args.n_nets)+'_'+args.data
        temp = torch.load(path)
        logall['ours_ot']['acc'] += temp['ours_ot']['acc']
        logall['ours']['acc'] += temp['ours']['acc']
        logall['ensemble_acc'] += temp['ensemble_acc']
        logall['ot_acc'] += temp['ot_acc']
        logall['fedavg_acc'] += temp['fedavg_acc']
    logall['ours_ot']['acc'] /= args.repeat
    logall['ours']['acc'] /= args.repeat
    logall['ensemble_acc'] /= args.repeat
    logall['ot_acc'] /= args.repeat
    logall['fedavg_acc'] /= args.repeat
    path='./'+args.logdir+'/'+str(args.diff_init)+'_'+args.expe+'_logdata_all_'+str(args.n_nets)+'_'+args.data
    torch.save(logall,path)
