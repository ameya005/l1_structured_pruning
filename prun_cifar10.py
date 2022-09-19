import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

#from cifar_resnet import resnet18_in
import cifar_2 as resnet
from cifar_2 import ResNet18

import torch_pruning as tp
import argparse
import torch
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn 
import numpy as np

from cifar_resnet import LearnableAlpha, resnet18_in


# class MyPruningFn(tp.prune.structured.BasePruningFunction):
#     @staticmethod
#     def prune_params(layer: LearnableAlpha, idxs: Sequence[int]) -> nn.Module: 
#         keep_idxs = list(set(range(layer.in_dim)) - set(idxs))
#         layer.in_dim = layer.in_dim-len(idxs)
#         layer.scale = torch.nn.Parameter(layer.scale.data.clone()[keep_idxs])
#         layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
#         return layer
    
#     @staticmethod
#     def calc_nparams_to_prune(layer: LearnableAlpha, idxs: Sequence[int]) -> int: 
#         nparams_to_prune = len(idxs) * 2
#         return nparams_to_prune

# # function wrapper
# def my_pruning_fn(layer: LearnableAlpha, idxs: list, inplace: bool=True, dry_run: bool=False, **kwargs):
#     return MyPruningFn.apply(layer, idxs, inplace, dry_run, **kwargs)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--step_size', type=int, default=70)
parser.add_argument('--round', type=int, default=1)
parser.add_argument('-mpath', type=str, default='')
parser.add_argument('-stride', type=int, default=1)
parser.add_argument('--outdir', type=str, default='./')


args = parser.parse_args()
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

def get_dataloader():
    if args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            CIFAR10('./data', train=True, transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]), download=True),batch_size=args.batch_size, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]),download=True),batch_size=args.batch_size, num_workers=2)
    elif args.dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            CIFAR100('./data', train=True, transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]), download=True),batch_size=args.batch_size, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            CIFAR100('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]),download=True),batch_size=args.batch_size, num_workers=2)


    return train_loader, test_loader

def eval(model, test_loader):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred==target).sum()
            total += len(target)
    return correct / total

def train_model(model, train_loader, test_loader):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)
    model.to(device)

    best_acc = -1
    for epoch in range(args.total_epochs):
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            if i%10==0 and args.verbose:
                print("Epoch %d/%d, iter %d/%d, loss=%.4f"%(epoch, args.total_epochs, i, len(train_loader), loss.item()))
        model.eval()
        acc = eval(model, test_loader)
        print("Epoch %d/%d, Acc=%.4f"%(epoch, args.total_epochs, acc))
        if best_acc<acc:
            torch.save( model, args.outdir + '/finetune_model.pth' )
            best_acc=acc
        scheduler.step()
    print("Best Acc=%.4f"%(best_acc))

def prune_model(model):
    #model.cpu()
    #print(model)
    DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, 32, 32) )
    def prune_conv(conv, amount=0.2):
        #weight = conv.weight.detach().cpu().numpy()
        #out_channels = weight.shape[0]
        #L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
        #num_pruned = int(out_channels * pruned_prob)
        #pruning_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
        strategy = tp.strategy.L1Strategy()
        pruning_index = strategy(conv.weight, amount=amount)
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()
    
    #block_prune_probs = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    #block_prune_probs = [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
    #block_prune_probs = [0.9, 0.9, 0.9, 0.7, 0.7, 0.7, 0.6, 0.6]
    #block_prune_probs = [0.65, 0.65, 0.65, 0.65, 0.6, 0.6, 0.6, 0.6]
    #block_prune_probs = [0.65, 0.65, 0.65, 0.65, 0.5, 0.5, 0.5, 0.5]
    block_prune_probs = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    blk_id = 0
    for m in model.modules():
        if isinstance( m, resnet.BasicBlock ):
            prune_conv( m.conv1, block_prune_probs[blk_id] )
            prune_conv( m.conv2, block_prune_probs[blk_id] )
            blk_id+=1
    return model    

def main():
    train_loader, test_loader = get_dataloader()
    if args.mode=='train':
        args.round=0
        model = ResNet18(num_classes=10 if args.dataset=='cifar10' else 100)
        train_model(model, train_loader, test_loader)
    elif args.mode=='prune':
        print('in prune')
        model = ResNet18(num_classes=10 if args.dataset=='cifar10' else 100)
        previous_ckpt = torch.load(args.mpath)
        print("Pruning ")

        #sd = previous_ckpt['state_dict']
        # for i in sd.keys():
        #     if 'alphas' in i:
        #         _,ch,_,_ = sd[i].shape
    #         sd[i] = torch.ones(1,ch,1,1)

        #model = model.load_state_dict(sd, strict=False)
        #print(model)
        prune_model(model)
        #print(model)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM"%(params/1e6))
        torch.save(model, args.outdir +'/pruned_model.pth')
        train_model(model, train_loader, test_loader)
    elif args.mode=='test':
        ckpt = args.mpath
        #model = torch.load(ckpt)
        #print("Load model from %s"%( ckpt ))
        model = torch.load( ckpt )
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM"%(params/1e6))
        acc = eval(model, test_loader)
        print("Acc=%.4f\n"%(acc)) 

if __name__=='__main__':
    main()
