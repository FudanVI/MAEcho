'''
we use OT to test the effect of MAPCO+neural_matching_based method.
code from https://github.com/sidak/otfusion
'''
import argparse
import json
import copy
import os
import torch
from wasserstein_ensemble import geometric_ensembling_modularized
from wasserstein_ensemble import get_wasserstein
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', default=1, type=int, help='GPU id to use')
    parser.add_argument('--skip-last-layer', default=True, help='skip the last layer in calculating optimal transport')
    parser.add_argument('--skip-last-layer-type', type=str, default='average', choices=['second', 'average'],
                        help='how to average the parameters for the last layer')
    parser.add_argument('--debug', default=False, help='print debug statements')
    parser.add_argument('--reg', default=1e-2, type=float, help='regularization strength for sinkhorn (default: 1e-2)')
    parser.add_argument('--reg-m', default=1e-3, type=float, help='regularization strength for marginals in unbalanced sinkhorn (default: 1e-3)')
    parser.add_argument('--ground-metric', type=str, default='euclidean', choices=['euclidean', 'cosine'],
                        help='ground metric for OT calculations, only works in free support v2 and soon with Ground Metric class in all! .')
    parser.add_argument('--ground-metric-normalize', type=str, default='none', choices=['log', 'max', 'none', 'median', 'mean'],
                        help='ground metric normalization to consider! ')
    parser.add_argument('--not-squared', default=True, help='dont square the ground metric')
    parser.add_argument('--clip-gm', default=False, help='to clip ground metric')
    parser.add_argument('--clip-min', type=float, default=0,
                       help='Value for clip-min for gm')
    parser.add_argument('--clip-max', type=float, default=5,
                       help='Value for clip-max for gm')
    parser.add_argument('--tmap-stats', default=False, help='print tmap stats')
    parser.add_argument('--ensemble-step', type=float, default=0.5, action='store', help='rate of adjustment towards the second model')
    parser.add_argument('--ground-metric-eff', default=True, help='memory efficient calculation of ground metric')
    parser.add_argument('--eval-aligned', action='store_true',
                        help='evaluate the accuracy of the aligned model 0')
    parser.add_argument('--correction',default=True, help='scaling correction for OT')
#     parser.add_argument('--weight-stats', default=True, help='log neuron-wise weight vector stats.')
    parser.add_argument('--sinkhorn-type', type=str, default='normal', choices=['normal', 'stabilized', 'epsilon', 'gpu'],
                        help='Type of sinkhorn algorithm to consider.')
    parser.add_argument('--geom-ensemble-type', type=str, default='wts', choices=['wts', 'acts'],
                        help='Ensemble based on weights (wts) or activations (acts).')

    parser.add_argument('--normalize-wts', default=True,
                        help='normalize the vector of weights')
    parser.add_argument('--gromov', default=False, help='use gromov wasserstein distance and barycenters')
    parser.add_argument('--gromov-loss', type=str, default='square_loss',
                        choices=['square_loss', 'kl_loss'], help="choice of loss function for gromov wasserstein computations")
    parser.add_argument('--past-correction', default=True, help='use the current weights aligned by multiplying with past transport map')
    parser.add_argument('--exact', default=True, help='compute exact optimal transport')
    parser.add_argument('--proper-marginals', default=False, help='consider the marginals of transport map properly')
    parser.add_argument('--print-distances', default=True, help='print OT distances for every layer')
    parser.add_argument('--importance', type=str, default=None,
                        help='importance measure to use for building probab mass! (options, l1, l2, l11, l12)')
    parser.add_argument('--unbalanced', default=False, help='use unbalanced OT')
    
    return parser


parser = get_parser()

base_args = parser.parse_known_args()[0]

def doot(net,model_type='mlpnet',args=None):
    '''
    net=[net1,net2]
    '''
    base_args.gpu_id = args.gpu_id
    modelout = geometric_ensembling_modularized(base_args, net,mtype=model_type)
    return modelout

