# README
The official implementation of 'One-shot Federated Learning without Server-side Training' in *Neural Networks 2023*.
## Requirements
```
python 3.7.3
pytorch 1.5.0
torchvision 0.6.0
tqdm 4.46.0
numpy 1.16.4
POT 0.7.0
```
## Sample Commands:
By default, the experiment contains a lot of testing processes, so it is time-consuming. If you only need to see the final accuracy of the aggregation, just add '--test False'.

Experiment MNIST+MLP
```
python noniid.py --alpha 0.01 --model_type mlpnet --data mnist --n_nets 2 --diff_init True --maxt_times 60 --C 0.5 --test True --lambdastep 1.6 --logdir logfinal --expe 5mlp
```

Experiment CIFAR+CNN
```
python noniid.py --alpha 0.01 --model_type cnnnet --data cifar --n_nets 2 --diff_init True --norm True --maxt_times 300 --C 0.5 --test True --lambdastep 0.05 --logdir logfinal --expe 5cnn
```
##Output:
The main information is recorded in the log file, including:

```
log = {
        'ours_ot':{
            'acc':ours_ot,},
        'ours':{
            'acc':ours,},
        'ensemble_acc':ensemble_acc,
        'ot_acc':ot_acc,
        'fedavg_acc':fedavg_acc,
        'all_model_acc':np.array(acc),
        'partition':traindata_cls_counts,
        }

```