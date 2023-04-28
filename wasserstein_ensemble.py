import ot
import torch
import numpy as np
# import routines
# from model import get_model_from_name
# import utils
from ground_metric import GroundMetric
import math
import sys
# import compute_activations
import modelset as mynetwork
torch.manual_seed(0)
torch.cuda.manual_seed(0) 
np.random.seed(0) 


def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    return c

def get_histogram(args, idx, cardinality, layer_name=None,  return_numpy = True, float64=False):
    
    if not args.unbalanced:
#             print("returns a uniform measure of cardinality: ", cardinality)
        return np.ones(cardinality)/cardinality
    else:
        return np.ones(cardinality)
    
def get_wassersteinized_layers_modularized(args, networks, eps=1e-7, test_loader=None,pp1=None):
    '''
    Two neural networks that have to be averaged in geometric manner (i.e. layerwise).
    The 1st network is aligned with respect to the other via wasserstein distance.
    Also this assumes that all the layers are either fully connected or convolutional *(with no bias)*

    :param networks: list of networks
    :param activations: If not None, use it to build the activation histograms.
    Otherwise assumes uniform distribution over neurons in a layer.
    :return: list of layer weights 'wassersteinized'
    '''

    # simple_model_0, simple_model_1 = networks[0], networks[1]
    # simple_model_0 = get_trained_model(0, model='simplenet')
    # simple_model_1 = get_trained_model(1, model='simplenet')

    avg_aligned_layers = []
    # cumulative_T_var = None
    T_var = None
    # print(list(networks[0].parameters()))
    previous_layer_shape = None
    ground_metric_object = GroundMetric(args)

    if args.eval_aligned:
        model0_aligned_layers = []

    if args.gpu_id==-1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu_id))


    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    for idx, ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) in \
            enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):
        if 'running' in layer0_name or 'batch' in layer0_name:continue
        assert fc_layer0_weight.shape == fc_layer1_weight.shape
#         print("Previous layer shape is ", previous_layer_shape)
        previous_layer_shape = fc_layer1_weight.shape

        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]

        # mu = np.ones(fc_layer0_weight.shape[0])/fc_layer0_weight.shape[0]
        # nu = np.ones(fc_layer1_weight.shape[0])/fc_layer1_weight.shape[0]

        layer_shape = fc_layer0_weight.shape
        if len(layer_shape) > 2:
            is_conv = True
            # For convolutional layers, it is (#out_channels, #in_channels, height, width)
            fc_layer0_weight_data = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
#             if pp1:
#                 pp1[idx] = pp1[idx].view(fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
            fc_layer1_weight_data = fc_layer1_weight.data.view(fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1)
        else:
            is_conv = False
            fc_layer0_weight_data = fc_layer0_weight.data
            fc_layer1_weight_data = fc_layer1_weight.data

        if idx == 0:
            if is_conv:
                M = ground_metric_object.process(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                                fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
                # M = cost_matrix(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                #                 fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
            else:
                # print("layer data is ", fc_layer0_weight_data, fc_layer1_weight_data)
                M = ground_metric_object.process(fc_layer0_weight_data, fc_layer1_weight_data)
                # M = cost_matrix(fc_layer0_weight, fc_layer1_weight)

            aligned_wt = fc_layer0_weight_data
        else:

#             print("shape of layer: model 0", fc_layer0_weight_data.shape)
#             print("shape of layer: model 1", fc_layer1_weight_data.shape)
#             print("shape of previous transport map", T_var.shape)

            # aligned_wt = None, this caches the tensor and causes OOM
            if is_conv:
                
                T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)# (#out_channels, #in_channels, height, width)
                aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)
#                 if pp1!=None:
                    
                
                M = ground_metric_object.process(
                    aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                    fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
                )
            else:
                if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                    # Handles the switch from convolutional layers to fc layers
                    fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0], -1).permute(2, 0, 1)
                    aligned_wt = torch.bmm(
                        fc_layer0_unflattened,
                        T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                    ).permute(1, 2, 0)
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                else:
                    # print("layer data (aligned) is ", aligned_wt, fc_layer1_weight_data)
                    aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)
                # M = cost_matrix(aligned_wt, fc_layer1_weight)
                M = ground_metric_object.process(aligned_wt, fc_layer1_weight)
#                 print("ground metric is ", M)
            if args.skip_last_layer and idx == (num_layers - 1):
#                 print("Simple averaging of last layer weights. NO transport map needs to be computed")
                if args.ensemble_step != 0.5:
                    avg_aligned_layers.append(aligned_wt)
#                     avg_aligned_layers.append((1 - args.ensemble_step) * aligned_wt +
#                                           args.ensemble_step * fc_layer1_weight)
                else:
                    avg_aligned_layers.append(aligned_wt)
#                     avg_aligned_layers.append((aligned_wt + fc_layer1_weight)/2)
#                 return avg_aligned_layers
                if pp1!=None:
                    return avg_aligned_layers,pp1
                else:
                    return avg_aligned_layers

        if args.importance is None or (idx == num_layers -1):
            mu = get_histogram(args, 0, mu_cardinality, layer0_name)
            nu = get_histogram(args, 1, nu_cardinality, layer1_name)
        else:
            # mu = _get_neuron_importance_histogram(args, aligned_wt, is_conv)
            mu = _get_neuron_importance_histogram(args, fc_layer0_weight_data, is_conv)
            nu = _get_neuron_importance_histogram(args, fc_layer1_weight_data, is_conv)
            print(mu, nu)
            assert args.proper_marginals

        cpuM = M.data.cpu().numpy()
        if args.exact:
            T = ot.emd(mu, nu, cpuM)
#             print("*******************************",mu.shape,nu.shape,cpuM.shape)
#             print(nu)
        else:
            T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=args.reg)
        # T = ot.emd(mu, nu, log_cpuM)
#         print("************",str(idx),"*****",T.shape)

        if args.gpu_id!=-1:
            T_var = torch.from_numpy(T).cuda(args.gpu_id).float()
#             if pp1:
#                 pp1[idx+1] = torch.bmm(pp1[idx+1], T_var_conv)
        else:
            T_var = torch.from_numpy(T).float()

        # torch.set_printoptions(profile="full")
#         print("the transport map is ", T_var)
        # torch.set_printoptions(profile="default")

        if args.correction:
            if not args.proper_marginals:
                # think of it as m x 1, scaling weights for m linear combinations of points in X
                if args.gpu_id != -1:
                    # marginals = torch.mv(T_var.t(), torch.ones(T_var.shape[0]).cuda(args.gpu_id))  # T.t().shape[1] = T.shape[0]
                    marginals = torch.ones(T_var.shape[0]).cuda(args.gpu_id) / T_var.shape[0]
                else:
                    # marginals = torch.mv(T_var.t(),
                    #                      torch.ones(T_var.shape[0]))  # T.t().shape[1] = T.shape[0]
                    marginals = torch.ones(T_var.shape[0]) / T_var.shape[0]
                marginals = torch.diag(1.0/(marginals + eps))  # take inverse
                T_var = torch.matmul(T_var, marginals)
            else:
                # marginals_alpha = T_var @ torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)
                marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)

                marginals = (1 / (marginals_beta + eps))
#                 print("shape of inverse marginals beta is ", marginals_beta.shape)
#                 print("inverse marginals beta is ", marginals_beta)

                T_var = T_var * marginals
                # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
                # this should all be ones, and number equal to number of neurons in 2nd model
                print(T_var.sum(dim=0))
                # assert (T_var.sum(dim=0) == torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)).all()

        if args.debug:
            if idx == (num_layers - 1):
                print("there goes the last transport map: \n ", T_var)
            else:
                print("there goes the transport map at layer {}: \n ".format(idx), T_var)

            print("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))

#         print("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))
#         print("Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var), torch.sum(T_var)))
        setattr(args, 'trace_sum_ratio_{}'.format(layer0_name), (torch.trace(T_var) / torch.sum(T_var)).item())

        if args.past_correction:
#             print("this is past correction for weight mode")
#             print("Shape of aligned wt is ", aligned_wt.shape)
#             print("Shape of fc_layer0_weight_data is ", fc_layer0_weight_data.shape)
            t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
            if pp1:
                pp1[idx+1] = torch.mm(pp1[idx+1],T_var)
#             print("******",str(idx),"**********",T_var.shape)
        else:
            t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))
            if pp1:
                pp1[idx+1] = torch.mm(pp1[idx+1],T_var)

        # Average the weights of aligned first layers
        
        if args.ensemble_step != 0.5:
            geometric_fc = t_fc0_model
#             geometric_fc = ((1-args.ensemble_step) * t_fc0_model +
#                             args.ensemble_step * fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
        else:
            geometric_fc = t_fc0_model
#             geometric_fc = (t_fc0_model + fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))/2
        if is_conv and layer_shape != geometric_fc.shape:
            geometric_fc = geometric_fc.view(layer_shape)
        avg_aligned_layers.append(geometric_fc)

        # get the performance of the model 0 aligned with respect to the model 1
#         if args.eval_aligned:
#             if is_conv and layer_shape != t_fc0_model.shape:
#                 t_fc0_model = t_fc0_model.view(layer_shape)
#             model0_aligned_layers.append(t_fc0_model)
#             _, acc = update_model(args, networks[0], model0_aligned_layers, test=True,
#                                   test_loader=test_loader, idx=0)
#             print("For layer idx {}, accuracy of the updated model is {}".format(idx, acc))
#             setattr(args, 'model0_aligned_acc_layer_{}'.format(str(idx)), acc)
#             if idx == (num_layers - 1):
#                 setattr(args, 'model0_aligned_acc', acc)
    if pp1!=None:
        return avg_aligned_layers,pp1
    else:
        return avg_aligned_layers

def get_wasserstein(args, fc_layer0_weight,fc_layer1_weight, eps=1e-7, test_loader=None,pp1=None):
   
    ground_metric_object = GroundMetric(args)
    assert fc_layer0_weight.shape == fc_layer1_weight.shape

    mu_cardinality = fc_layer0_weight.shape[0]
    nu_cardinality = fc_layer1_weight.shape[0]


    layer_shape = fc_layer0_weight.shape
    if len(layer_shape) > 2:
        is_conv = True
        fc_layer0_weight_data = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
        fc_layer1_weight_data = fc_layer1_weight.data.view(fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1)
    else:
        is_conv = False
        fc_layer0_weight_data = fc_layer0_weight.data
        fc_layer1_weight_data = fc_layer1_weight.data

    if is_conv:
        M = ground_metric_object.process(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                        fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
    else: 
        M = ground_metric_object.process(fc_layer0_weight_data, fc_layer1_weight_data)
        
    mu = get_histogram(args, 0, mu_cardinality)
    nu = get_histogram(args, 1, nu_cardinality)
    cpuM = M.data.cpu().numpy()
    if args.exact:
        dd = ot.emd2(mu, nu, cpuM)

    else:
        dd = ot.bregman.sinkhorn2(mu, nu, cpuM, reg=args.reg)

    return dd

# def print_stats(arr, nick=""):
#     print(nick)
#     print("summary stats are: \n max: {}, mean: {}, min: {}, median: {}, std: {} \n".format(
#         arr.max(), arr.mean(), arr.min(), np.median(arr), arr.std()
#     ))

# def get_activation_distance_stats(activations_0, activations_1, layer_name=""):
# #     if layer_name != "":
# #         print("In layer {}: getting activation distance statistics".format(layer_name))
#     M = cost_matrix(activations_0, activations_1) ** (1/2)
#     mean_dists =  torch.mean(M, dim=-1)
#     max_dists = torch.max(M, dim=-1)[0]
#     min_dists = torch.min(M, dim=-1)[0]
#     std_dists = torch.std(M, dim=-1)

#     print("Statistics of the distance from neurons of layer 1 (averaged across nodes of layer 0): \n")
#     print("Max : {}, Mean : {}, Min : {}, Std: {}".format(torch.mean(max_dists), torch.mean(mean_dists), torch.mean(min_dists), torch.mean(std_dists)))

# def update_model(args, model, new_params, test=False, test_loader=None, reversed=False, idx=-1):

#     updated_model = mynetwork.mnistnet()
#     if args.gpu_id != -1:
#         updated_model = updated_model.cuda(args.gpu_id)

#     layer_idx = 0
#     model_state_dict = model.state_dict()

# #     print("len of model_state_dict is ", len(model_state_dict.items()))
# #     print("len of new_params is ", len(new_params))

#     for key, value in model_state_dict.items():
# #         print("updated parameters for layer ", key)
#         model_state_dict[key] = new_params[layer_idx]
#         layer_idx += 1
#         if layer_idx == len(new_params):
#             break


#     updated_model.load_state_dict(model_state_dict)

# #     if test:
# #         log_dict = {}
# #         log_dict['test_losses'] = []
# #         final_acc = routines.test(args, updated_model, test_loader, log_dict)
# #         print("accuracy after update is ", final_acc)
# #     else:
# #          final_acc = None

#     return updated_model#, final_acc




def _get_layer_weights(layer_weight, is_conv):
    if is_conv:
        # For convolutional layers, it is (#out_channels, #in_channels, height, width)
        layer_weight_data = layer_weight.data.view(layer_weight.shape[0], layer_weight.shape[1], -1)
    else:
        layer_weight_data = layer_weight.data

    return layer_weight_data




def _custom_sinkhorn(args, mu, nu, cpuM):
    if not args.unbalanced:
        if args.sinkhorn_type == 'normal':
            T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=args.reg)
        elif args.sinkhorn_type == 'stabilized':
            T = ot.bregman.sinkhorn_stabilized(mu, nu, cpuM, reg=args.reg)
        elif args.sinkhorn_type == 'epsilon':
            T = ot.bregman.sinkhorn_epsilon_scaling(mu, nu, cpuM, reg=args.reg)
#         elif args.sinkhorn_type == 'gpu':
#             T, _ = utils.sinkhorn_loss(cpuM, mu, nu, gpu_id=args.gpu_id, epsilon=args.reg, return_tmap=True)
        else:
            raise NotImplementedError
    else:
        T = ot.unbalanced.sinkhorn_knopp_unbalanced(mu, nu, cpuM, reg=args.reg, reg_m=args.reg_m)
    return T


def _sanity_check_tmap(T):
    if not math.isclose(np.sum(T), 1.0, abs_tol=1e-7):
#         print("Sum of transport map is ", np.sum(T))
        raise Exception('NAN inside Transport MAP. Most likely due to large ground metric values')



# def _check_layer_sizes(args, layer_idx, shape1, shape2, num_layers):
#     if args.width_ratio == 1:
#         return shape1 == shape2
#     else:
#         if args.dataset == 'mnist':
#             if layer_idx == 0:
#                 return shape1[-1] == shape2[-1] and (shape1[0]/shape2[0]) == args.width_ratio
#             elif layer_idx == (num_layers -1):
#                 return (shape1[-1]/shape2[-1]) == args.width_ratio and shape1[0] == shape2[0]
#             else:
#                 ans = True
#                 for ix in range(len(shape1)):
#                     ans = ans and shape1[ix]/shape2[ix] == args.width_ratio
#                 return ans
#         elif args.dataset[0:7] == 'Cifar10':
#             assert args.second_model_name is not None
#             if layer_idx == 0 or layer_idx == (num_layers -1):
#                 return shape1 == shape2
#             else:
#                 if (not args.reverse and layer_idx == (num_layers-2)) or (args.reverse and layer_idx == 1):
#                     return (shape1[1] / shape2[1]) == args.width_ratio
#                 else:
#                     return (shape1[0]/shape2[0]) == args.width_ratio


def _compute_marginals(args, T_var, device, eps=1e-7):
    if args.correction:
        if not args.proper_marginals:
            # think of it as m x 1, scaling weights for m linear combinations of points in X
            marginals = torch.ones(T_var.shape)
            if args.gpu_id != -1:
                marginals = marginals.cuda(args.gpu_id)

            marginals = torch.matmul(T_var, marginals)
            marginals = 1 / (marginals + eps)
#             print("marginals are ", marginals)

            T_var = T_var * marginals

        else:
            # marginals_alpha = T_var @ torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)
            marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)

            marginals = (1 / (marginals_beta + eps))
#             print("shape of inverse marginals beta is ", marginals_beta.shape)
#             print("inverse marginals beta is ", marginals_beta)

            T_var = T_var * marginals
            # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
            # this should all be ones, and number equal to number of neurons in 2nd model
            print(T_var.sum(dim=0))
            # assert (T_var.sum(dim=0) == torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)).all()

#         print("T_var after correction ", T_var)
#         print("T_var stats: max {}, min {}, mean {}, std {} ".format(T_var.max(), T_var.min(), T_var.mean(),
#                                                                      T_var.std()))
    else:
        marginals = None

    return T_var, marginals

def _get_current_layer_transport_map(args, mu, nu, M0, M1, idx, layer_shape, eps=1e-7, layer_name=None):

    if not args.gromov:
        cpuM = M0.data.cpu().numpy()
        if args.exact:
            T = ot.emd(mu, nu, cpuM)
        else:
            T = _custom_sinkhorn(args, mu, nu, cpuM)

        if args.print_distances:
            ot_cost = np.multiply(T, cpuM).sum()
#             print(f'At layer idx {idx} and shape {layer_shape}, the OT cost is ', ot_cost)
            if layer_name is not None:
                setattr(args, f'{layer_name}_layer_{idx}_cost', ot_cost)
            else:
                setattr(args, f'layer_{idx}_cost', ot_cost)
    else:
        cpuM0 = M0.data.cpu().numpy()
        cpuM1 = M1.data.cpu().numpy()

        assert not args.exact
        T = ot.gromov.entropic_gromov_wasserstein(cpuM0, cpuM1, mu, nu, loss_fun=args.gromov_loss, epsilon=args.reg)

    if not args.unbalanced:
        _sanity_check_tmap(T)

    if args.gpu_id != -1:
        T_var = torch.from_numpy(T).cuda(args.gpu_id).float()
    else:
        T_var = torch.from_numpy(T).float()

#     if args.tmap_stats:
#         print(
#         "Tmap stats (before correction) \n: For layer {}, frobenius norm from the joe's transport map is {}".format(
#             layer0_name, torch.norm(T_var - torch.ones_like(T_var) / torch.numel(T_var), p='fro')
#         ))

#     print("shape of T_var is ", T_var.shape)
#     print("T_var before correction ", T_var)

    return T_var

def _get_neuron_importance_histogram(args, layer_weight, is_conv, eps=1e-9):
#     print('shape of layer_weight is ', layer_weight.shape)
    if is_conv:
        layer = layer_weight.contiguous().view(layer_weight.shape[0], -1).cpu().numpy()
    else:
        layer = layer_weight.cpu().numpy()
    
    if args.importance == 'l1':
        importance_hist = np.linalg.norm(layer, ord=1, axis=-1).astype(
                    np.float64) + eps
    elif args.importance == 'l2':
        importance_hist = np.linalg.norm(layer, ord=2, axis=-1).astype(
                    np.float64) + eps
    else:
        raise NotImplementedError

    if not args.unbalanced:
        importance_hist = (importance_hist/importance_hist.sum())
#         print('sum of importance hist is ', importance_hist.sum())
    # assert importance_hist.sum() == 1.0
    return importance_hist


def get_network_from_param_list(args, param_list, mtype,test_loader):

#     print("using independent method")
    if mtype=='mlpnet':
        new_network = mynetwork.mnistnet()#get_model_from_name(args, idx=1)
    elif mtype=='cnnnet':
        new_network = mynetwork.cnnNet()#get_model_from_name(args, idx=1)
    if args.gpu_id != -1:
        new_network = new_network.cuda(args.gpu_id)

    # check the test performance of the network before
#     log_dict = {}
#     log_dict['test_losses'] = []
#     routines.test(args, new_network, test_loader, log_dict)

    # set the weights of the new network
    # print("before", new_network.state_dict())
#     print("len of model parameters and avg aligned layers is ", len(list(new_network.parameters())),
#           len(param_list))
    assert len(list(new_network.parameters())) == len(param_list)

    layer_idx = 0
    model_state_dict = new_network.state_dict()

#     print("len of model_state_dict is ", len(model_state_dict.items()))
#     print("len of param_list is ", len(param_list))

    for key, value in model_state_dict.items():
        model_state_dict[key] = param_list[layer_idx]
        layer_idx += 1

    new_network.load_state_dict(model_state_dict)

    # check the test performance of the network after
#     log_dict = {}
#     log_dict['test_losses'] = []
#     acc = routines.test(args, new_network, test_loader, log_dict)

    return  new_network

def geometric_ensembling_modularized(args, networks, train_loader=None,mtype='mlpnet' ,test_loader=None,pp1=None):
    
    if pp1!=None:
        avg_aligned_layers,pp1 = get_wassersteinized_layers_modularized(args, networks, test_loader=test_loader,pp1=pp1)
        mmm = get_network_from_param_list(args, avg_aligned_layers,mtype, test_loader)
        return mmm,pp1
    else:
        avg_aligned_layers = get_wassersteinized_layers_modularized(args, networks, test_loader=test_loader)
        return get_network_from_param_list(args, avg_aligned_layers,mtype, test_loader)
    

