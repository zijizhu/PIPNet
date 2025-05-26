from util.log import Log
import torch.nn as nn
from util.args import get_args, save_args

from util.func import init_weights_xavier
# from pipnet.train import train_pipnet
from pipnet.test import eval_pipnet
import torch
from util.vis_pipnet import visualize, visualize_topk
import sys, os
import random
import numpy as np
import matplotlib.pyplot as plt
from v1 import get_cub_dataloaders, create_model, get_optimizer_nn, calculate_loss
from tqdm import tqdm


"""
For batch 2 testing
"""
def train_pipnet(net, train_loader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion, epoch, nr_epochs, device, pretrain=False, finetune=False, progress_prefix: str = 'Train Epoch'):

    # Make sure the model is in train mode
    net.train()
    
    if pretrain:
        # Disable training of classification layer
        net.module._classification.requires_grad = False
        progress_prefix = 'Pretrain Epoch'
    else:
        # Enable training of classification layer (disabled in case of pretraining)
        net.module._classification.requires_grad = True
    
    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.

    iters = len(train_loader)
    # Show progress on progress bar. 
    train_iter = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=progress_prefix+'%s'%epoch,
                    mininterval=2.,
                    ncols=0)
    
    count_param=0
    for name, param in net.named_parameters():
        if param.requires_grad:
            count_param+=1           
    print("Number of parameters that require gradient: ", count_param, flush=True)

    if pretrain:
        align_pf_weight = (epoch/nr_epochs)*1.
        unif_weight = 0.5 #ignored
        t_weight = 5.
        cl_weight = 0.
    else:
        align_pf_weight = 5. 
        t_weight = 2.
        unif_weight = 0.
        cl_weight = 2.

    
    print("Align weight: ", align_pf_weight, ", U_tanh weight: ", t_weight, "Class weight:", cl_weight, flush=True)
    print("Pretrain?", pretrain, "Finetune?", finetune, flush=True)
    
    lrs_net = []
    lrs_class = []
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs1, xs2, ys) in train_iter:       
        
        xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)
       
        # Reset the gradients
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)
       
        # Perform a forward pass through the network
        proto_features, pooled, out = net(torch.cat([xs1, xs2]))
        loss, acc = calculate_loss(proto_features, pooled, out, ys,
                                   align_pf_weight, t_weight, cl_weight, net_normalization_multiplier=net.module._classification.normalization_multiplier,
                                   pretrain=pretrain, finetune=finetune, criterion=criterion, train_iter=train_iter, print=True, EPS=1e-8)
        
        # Compute the gradient
        loss.backward()

        if not pretrain:
            optimizer_classifier.step()   
            scheduler_classifier.step(epoch - 1 + (i/iters))
            lrs_class.append(scheduler_classifier.get_last_lr()[0])
     
        if not finetune:
            optimizer_net.step()
            scheduler_net.step() 
            lrs_net.append(scheduler_net.get_last_lr()[0])
        else:
            lrs_net.append(0.)
            
        with torch.no_grad():
            total_acc+=acc
            total_loss+=loss.item()

        if not pretrain:
            with torch.no_grad():
                net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 1e-3, min=0.)) #set weights in classification layer < 1e-3 to zero
                net.module._classification.normalization_multiplier.copy_(torch.clamp(net.module._classification.normalization_multiplier.data, min=1.0)) 
                if net.module._classification.bias is not None:
                    net.module._classification.bias.copy_(torch.clamp(net.module._classification.bias.data, min=0.))  
    train_info['train_accuracy'] = total_acc/float(i+1)
    train_info['loss'] = total_loss/float(i+1)
    train_info['lrs_net'] = lrs_net
    train_info['lrs_class'] = lrs_class
    
    return train_info


def run_pipnet(args=None):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    args = args or get_args()
    assert args.batch_size > 1

    # Create a logger
    log = Log(args.log_dir)
    print("Log dir: ", args.log_dir, flush=True)
    # Log the run arguments
    save_args(args, log.metadata_dir)
    
    gpu_list = args.gpu_ids.split(',')
    device_ids = []
    if args.gpu_ids!='':
        for m in range(len(gpu_list)):
            device_ids.append(int(gpu_list[m]))
    
    global device
    if not args.disable_cuda and torch.cuda.is_available():
        if len(device_ids)==1:
            device = torch.device('cuda:{}'.format(args.gpu_ids))
        elif len(device_ids)==0:
            device = torch.device('cuda')
            print("CUDA device set without id specification", flush=True)
            device_ids.append(torch.cuda.current_device())
        else:
            print("This code should work with multiple GPU's but we didn't test that, so we recommend to use only 1 GPU.", flush=True)
            device_str = ''
            for d in device_ids:
                device_str+=str(d)
                device_str+=","
            device = torch.device('cuda:'+str(device_ids[0]))
    else:
        device = torch.device('cpu')
     
    # Log which device was actually used
    print("Device used: ", device, "with id", device_ids, flush=True)
    
    # Obtain the dataset and dataloaders

    trainloader, trainloader_pretraining, testloader, projectloader, classes = get_cub_dataloaders(batch_size=args.batch_size, batch_size_pretrain=args.batch_size_pretrain)
    if len(classes)<=20:
        if args.validation_size == 0.:
            print("Classes: ", testloader.dataset.class_to_idx, flush=True)
        else:
            print("Classes: ", str(classes), flush=True)
    
    # Create a convolutional network based on arguments and add 1x1 conv layer
    net = create_model()  # NOTE replaced

    num_prototypes = 768
    # Create a PIP-Net
    net = net.to(device=device)
    net = nn.DataParallel(net, device_ids = device_ids)    
    
    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net, args.lr_net, args.lr_block, args.lr)  # NOTE: replaced


    # Initialize or load model
    with torch.no_grad():
        if args.state_dict_dir_net != '':
            epoch = 0
            checkpoint = torch.load(args.state_dict_dir_net,map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'],strict=True) 
            print("Pretrained network loaded", flush=True)
            net.module._multiplier.requires_grad = False
            try:
                optimizer_net.load_state_dict(checkpoint['optimizer_net_state_dict']) 
            except:
                pass
            if torch.mean(net.module._classification.weight).item() > 1.0 and torch.mean(net.module._classification.weight).item() < 3.0 and torch.count_nonzero(torch.relu(net.module._classification.weight-1e-5)).float().item() > 0.8*(num_prototypes*len(classes)): #assume that the linear classification layer is not yet trained (e.g. when loading a pretrained backbone only)
                print("We assume that the classification layer is not yet trained. We re-initialize it...", flush=True)
                torch.nn.init.normal_(net.module._classification.weight, mean=1.0,std=0.1) 
                torch.nn.init.constant_(net.module._multiplier, val=2.)
                print("Classification layer initialized with mean", torch.mean(net.module._classification.weight).item(), flush=True)
                if args.bias:
                    torch.nn.init.constant_(net.module._classification.bias, val=0.)
            # else: #uncomment these lines if you want to load the optimizer too
            #     if 'optimizer_classifier_state_dict' in checkpoint.keys():
            #         optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier_state_dict'])
            
        else:
            net.module._add_on.apply(init_weights_xavier)
            torch.nn.init.normal_(net.module._classification.weight, mean=1.0,std=0.1) 
            if args.bias:
                torch.nn.init.constant_(net.module._classification.bias, val=0.)
            torch.nn.init.constant_(net.module._multiplier, val=2.)
            net.module._multiplier.requires_grad = False

            print("Classification layer initialized with mean", torch.mean(net.module._classification.weight).item(), flush=True)
    
    # Define classification loss function and scheduler
    criterion = nn.NLLLoss(reduction='mean').to(device)
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net, T_max=len(trainloader_pretraining)*args.epochs_pretrain, eta_min=args.lr_block/100., last_epoch=-1)

    # Forward one batch through the backbone to get the latent output size
    with torch.no_grad():
        xs1, _, _ = next(iter(trainloader))
        xs1 = xs1.to(device)
        proto_features, _, _ = net(xs1)
        wshape = proto_features.shape[-1]
        args.wshape = wshape #needed for calculating image patch size
        print("Output shape: ", proto_features.shape, flush=True)
    
    if net.module._num_classes == 2:
        # Create a csv log for storing the test accuracy, F1-score, mean train accuracy and mean loss for each epoch
        log.create_log('log_epoch_overview', 'epoch', 'test_top1_acc', 'test_f1', 'almost_sim_nonzeros', 'local_size_all_classes','almost_nonzeros_pooled', 'num_nonzero_prototypes', 'mean_train_acc', 'mean_train_loss_during_epoch')
        print("Your dataset only has two classes. Is the number of samples per class similar? If the data is imbalanced, we recommend to use the --weighted_loss flag to account for the imbalance.", flush=True)
    else:
        # Create a csv log for storing the test accuracy (top 1 and top 5), mean train accuracy and mean loss for each epoch
        log.create_log('log_epoch_overview', 'epoch', 'test_top1_acc', 'test_top5_acc', 'almost_sim_nonzeros', 'local_size_all_classes','almost_nonzeros_pooled', 'num_nonzero_prototypes', 'mean_train_acc', 'mean_train_loss_during_epoch')
    
    
    lrs_pretrain_net = []
    # PRETRAINING PROTOTYPES PHASE
    for epoch in range(1, args.epochs_pretrain+1):
        for param in params_to_train:
            param.requires_grad = True
        for param in net.module._add_on.parameters():
            param.requires_grad = True
        for param in net.module._classification.parameters():
            param.requires_grad = False
        for param in params_to_freeze:
            param.requires_grad = True # can be set to False when you want to freeze more layers
        for param in params_backbone:
            param.requires_grad = False #can be set to True when you want to train whole backbone (e.g. if dataset is very different from ImageNet)
        
        print("\nPretrain Epoch", epoch, "with batch size", trainloader_pretraining.batch_size, flush=True)
        
        # Pretrain prototypes
        # TODO: change this
        train_info = train_pipnet(net, trainloader_pretraining, optimizer_net, optimizer_classifier, scheduler_net, None, criterion, epoch, args.epochs_pretrain, device, pretrain=True, finetune=False)
        lrs_pretrain_net+=train_info['lrs_net']
        plt.clf()
        plt.plot(lrs_pretrain_net)
        plt.savefig(os.path.join(args.log_dir,'lr_pretrain_net.png'))
        log.log_values('log_epoch_overview', epoch, "n.a.", "n.a.", "n.a.", "n.a.", "n.a.", "n.a.", "n.a.", train_info['loss'])
    
    if args.state_dict_dir_net == '':
        net.eval()
        torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_pretrained'))
        net.train()
    with torch.no_grad():
        if 'convnext' in args.net and args.epochs_pretrain > 0:
            topks = visualize_topk(net, projectloader, len(classes), device, 'visualised_pretrained_prototypes_topk', args)
        
    # SECOND TRAINING PHASE
    # re-initialize optimizers and schedulers for second training phase
    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net, args.lr_net, args.lr_block, args.lr)  # NOTE: replaced
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net, T_max=len(trainloader)*args.epochs, eta_min=args.lr_net/100.)
    # scheduler for the classification layer is with restarts, such that the model can re-active zeroed-out prototypes. Hence an intuitive choice. 
    if args.epochs<=30:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifier, T_0=5, eta_min=0.001, T_mult=1, verbose=False)
    else:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifier, T_0=10, eta_min=0.001, T_mult=1, verbose=False)
    for param in net.module.parameters():
        param.requires_grad = False
    for param in net.module._classification.parameters():
        param.requires_grad = True
    
    frozen = True
    lrs_net = []
    lrs_classifier = []
   
    for epoch in range(1, args.epochs + 1):                      
        epochs_to_finetune = 3 #during finetuning, only train classification layer and freeze rest. usually done for a few epochs (at least 1, more depends on size of dataset)
        if epoch <= epochs_to_finetune and (args.epochs_pretrain > 0 or args.state_dict_dir_net != ''):
            for param in net.module._add_on.parameters():
                param.requires_grad = False
            for param in params_to_train:
                param.requires_grad = False
            for param in params_to_freeze:
                param.requires_grad = False
            for param in params_backbone:
                param.requires_grad = False
            finetune = True
        
        else: 
            finetune=False          
            if frozen:
                # unfreeze backbone
                if epoch>(args.freeze_epochs):
                    for param in net.module._add_on.parameters():
                        param.requires_grad = True
                    for param in params_to_freeze:
                        param.requires_grad = True
                    for param in params_to_train:
                        param.requires_grad = True
                    for param in params_backbone:
                        param.requires_grad = True   
                    frozen = False
                # freeze first layers of backbone, train rest
                else:
                    for param in params_to_freeze:
                        param.requires_grad = True #Can be set to False if you want to train fewer layers of backbone
                    for param in net.module._add_on.parameters():
                        param.requires_grad = True
                    for param in params_to_train:
                        param.requires_grad = True
                    for param in params_backbone:
                        param.requires_grad = False
        
        print("\n Epoch", epoch, "frozen:", frozen, flush=True)            
        if (epoch==args.epochs or epoch%30==0) and args.epochs>1:
            # SET SMALL WEIGHTS TO ZERO
            with torch.no_grad():
                torch.set_printoptions(profile="full")
                net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 0.001, min=0.)) 
                print("Classifier weights: ", net.module._classification.weight[net.module._classification.weight.nonzero(as_tuple=True)], (net.module._classification.weight[net.module._classification.weight.nonzero(as_tuple=True)]).shape, flush=True)
                if args.bias:
                    print("Classifier bias: ", net.module._classification.bias, flush=True)
                torch.set_printoptions(profile="default")

        train_info = train_pipnet(net, trainloader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion, epoch, args.epochs, device, pretrain=False, finetune=finetune)
        lrs_net+=train_info['lrs_net']
        lrs_classifier+=train_info['lrs_class']
        # Evaluate model
        eval_info = eval_pipnet(net, testloader, epoch, device, log)
        log.log_values('log_epoch_overview', epoch, eval_info['top1_accuracy'], eval_info['top5_accuracy'], eval_info['almost_sim_nonzeros'], eval_info['local_size_all_classes'], eval_info['almost_nonzeros'], eval_info['num non-zero prototypes'], train_info['train_accuracy'], train_info['loss'])
            
        with torch.no_grad():
            net.eval()
            torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict(), 'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained'))

            if epoch%30 == 0:
                net.eval()
                torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict(), 'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained_%s'%str(epoch)))            
        
            # save learning rate in figure
            plt.clf()
            plt.plot(lrs_net)
            plt.savefig(os.path.join(args.log_dir,'lr_net.png'))
            plt.clf()
            plt.plot(lrs_classifier)
            plt.savefig(os.path.join(args.log_dir,'lr_class.png'))
                
    net.eval()
    torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict(), 'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained_last'))

    topks = visualize_topk(net, projectloader, len(classes), device, 'visualised_prototypes_topk', args)
    # set weights of prototypes that are never really found in projection set to 0
    set_to_zero = []
    if topks:
        for prot in topks.keys():
            found = False
            for (i_id, score) in topks[prot]:
                if score > 0.1:
                    found = True
            if not found:
                torch.nn.init.zeros_(net.module._classification.weight[:,prot])
                set_to_zero.append(prot)
        print("Weights of prototypes", set_to_zero, "are set to zero because it is never detected with similarity>0.1 in the training set", flush=True)
        eval_info = eval_pipnet(net, testloader, "notused"+str(args.epochs), device, log)
        log.log_values('log_epoch_overview', "notused"+str(args.epochs), eval_info['top1_accuracy'], eval_info['top5_accuracy'], eval_info['almost_sim_nonzeros'], eval_info['local_size_all_classes'], eval_info['almost_nonzeros'], eval_info['num non-zero prototypes'], "n.a.", "n.a.")

    print("classifier weights: ", net.module._classification.weight, flush=True)
    print("Classifier weights nonzero: ", net.module._classification.weight[net.module._classification.weight.nonzero(as_tuple=True)], (net.module._classification.weight[net.module._classification.weight.nonzero(as_tuple=True)]).shape, flush=True)
    print("Classifier bias: ", net.module._classification.bias, flush=True)
    # Print weights and relevant prototypes per class
    for c in range(net.module._classification.weight.shape[0]):
        relevant_ps = []
        proto_weights = net.module._classification.weight[c,:]
        for p in range(net.module._classification.weight.shape[1]):
            if proto_weights[p]> 1e-3:
                relevant_ps.append((p, proto_weights[p].item()))
        if args.validation_size == 0.:
            print("Class", c, "(", list(testloader.dataset.class_to_idx.keys())[list(testloader.dataset.class_to_idx.values()).index(c)],"):","has", len(relevant_ps),"relevant prototypes: ", relevant_ps, flush=True)

    # Evaluate prototype purity        
    # if args.dataset == 'CUB-200-2011':
    #     projectset_img0_path = projectloader.dataset.samples[0][0]
    #     project_path = os.path.split(os.path.split(projectset_img0_path)[0])[0].split("dataset")[0]
    #     parts_loc_path = os.path.join(project_path, "parts/part_locs.txt")
    #     parts_name_path = os.path.join(project_path, "parts/parts.txt")
    #     imgs_id_path = os.path.join(project_path, "images.txt")
    #     cubthreshold = 0.5 

    #     net.eval()
    #     print("\n\nEvaluating cub prototypes for training set", flush=True)        
    #     csvfile_topk = get_topk_cub(net, projectloader, 10, 'train_'+str(epoch), device, args)
    #     eval_prototypes_cub_parts_csv(csvfile_topk, parts_loc_path, parts_name_path, imgs_id_path, 'train_topk_'+str(epoch), args, log)
        
    #     csvfile_all = get_proto_patches_cub(net, projectloader, 'train_all_'+str(epoch), device, args, threshold=cubthreshold)
    #     eval_prototypes_cub_parts_csv(csvfile_all, parts_loc_path, parts_name_path, imgs_id_path, 'train_all_thres'+str(cubthreshold)+'_'+str(epoch), args, log)
        
    #     print("\n\nEvaluating cub prototypes for test set", flush=True)
    #     csvfile_topk = get_topk_cub(net, test_projectloader, 10, 'test_'+str(epoch), device, args)
    #     eval_prototypes_cub_parts_csv(csvfile_topk, parts_loc_path, parts_name_path, imgs_id_path, 'test_topk_'+str(epoch), args, log)
    #     cubthreshold = 0.5
    #     csvfile_all = get_proto_patches_cub(net, test_projectloader, 'test_'+str(epoch), device, args, threshold=cubthreshold)
    #     eval_prototypes_cub_parts_csv(csvfile_all, parts_loc_path, parts_name_path, imgs_id_path, 'test_all_thres'+str(cubthreshold)+'_'+str(epoch), args, log)
        
    # # visualize predictions 
    # visualize(net, projectloader, len(classes), device, 'visualised_prototypes', args)
    # testset_img0_path = test_projectloader.dataset.samples[0][0]
    # test_path = os.path.split(os.path.split(testset_img0_path)[0])[0]
    # vis_pred(net, test_path, classes, device, args) 
    # if args.extra_test_image_folder != '':
    #     if os.path.exists(args.extra_test_image_folder):   
    #         vis_pred_experiments(net, args.extra_test_image_folder, classes, device, args)

    print("Done!", flush=True)

if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print_dir = os.path.join(args.log_dir,'out.txt')
    tqdm_dir = os.path.join(args.log_dir,'tqdm.txt')
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = open(print_dir, 'w')
    sys.stderr = open(tqdm_dir, 'w')
    run_pipnet(args)
    
    sys.stdout.close()
    sys.stderr.close()
