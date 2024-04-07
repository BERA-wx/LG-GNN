from LG_GNN_revise.model.model import LG_GNN
from LG_GNN_revise.utils import accuracy, load_data, kolmogorov_smirnov
from LG_GNN_revise.contrastive_loss.SupervisedContrastiveLoss import SupConLoss

import torch
from torch import nn
import torch.optim as optim
import os
import glob
import time
import random
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fast_mode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GNN with sparse version or not.')
parser.add_argument('--seed', type=int, default=70, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden_gnn', type=int, default=[128, 64], help='Number of hidden units of GNN.')
parser.add_argument('--nb_heads', type=int, default=2, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--share_weights', action='store_true', default=True,
                    help='The same matrix will be applied to the source and the target node of every edge.')
parser.add_argument('--temp', type=float, default=[0.07, 0.007], help='Temperature of contrastive learning.')
parser.add_argument('--beta', type=float, default=[0.4, 0.6], help='Weight of contrastive/classification loss.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# AUC, KS
auc_list = []
ks_list = []

# Load data
# p: Lending1(0.85), Lending2(0.95), Lending3(0.9)
# batch_size: Lending1(1000), Lending2(1000), Lending3(2000)
datasets = load_data(dataset_name='Lending1', p=0.85, batch=1000)
for idx, batch in enumerate(datasets):
    adj = batch['adj']
    features = batch['features']
    labels = batch['labels']
    idx_train = batch['idx_train']
    idx_val = batch['idx_val']
    idx_test = batch['idx_test']

    model = LG_GNN(in_features=features.shape[1],
                   n_hidden_gnn=args.hidden_gnn,
                   n_classes=2,
                   n_heads=args.nb_heads,
                   k_l=0.1,  # Lending1(0.1), Lending2(0.35), Lending3(0.4)
                   k_g=0.9,  # Lending1(0.9), Lending2(0.85), Lending3(0.1)
                   dropout=args.dropout,
                   num_unique_values=32,
                   embed_dim=4,
                   share_weights=args.share_weights,
                   head='linear',
                   classifier='linear')
    # loss
    criterion_cl = SupConLoss(temp=args.temp)
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    # Cuda
    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()


    def train():
        model.train()
        optimizer.zero_grad()
        ce_feats, cl_feats = model(features, adj)
        # loss
        loss_train_cl = criterion_cl(cl_feats[idx_train], labels[idx_train])
        loss_train_mlp = criterion(ce_feats[idx_train], labels[idx_train])
        loss_train = args.beta * loss_train_cl + (1 - args.beta) * loss_train_mlp
        # acc
        acc_train = accuracy(ce_feats[idx_train], labels[idx_train])
        # Back Propagation
        loss_train.backward()
        optimizer.step()

        if not args.fast_mode:
            model.eval()
            ce_feats, cl_feats = model(features, adj)
        # loss
        loss_val_cl = criterion_cl(cl_feats[idx_val], labels[idx_val])
        loss_val_mlp = criterion(ce_feats[idx_val], labels[idx_val])
        loss_val = args.beta * loss_val_cl + (1 - args.beta) * loss_val_mlp
        acc_val = accuracy(ce_feats[idx_val], labels[idx_val])
        # print('Epoch: {:04d}'.format(epoch + 1),
        #       'loss_train: {:.4f}'.format(loss_train.data.item()),
        #       'acc_train: {:.4f}'.format(acc_train.data.item()),
        #       'loss_val: {:.4f}'.format(loss_val.data.item()),
        #       'acc_val: {:.4f}'.format(acc_val.data.item()))
        return loss_val.data.item()


    def compute_test():
        model.eval()
        ce_feats, cl_feats = model(features, adj)
        # loss
        loss_test_cl = criterion_cl(cl_feats[idx_test], labels[idx_test])
        loss_test_mlp = criterion(ce_feats[idx_test], labels[idx_test])
        loss_test = args.beta * loss_test_cl + (1 - args.beta) * loss_test_mlp

        # AUC
        auc_test = roc_auc_score(labels[idx_test].cpu().detach().numpy(),
                                 ce_feats[idx_test][:, 1].cpu().detach().numpy())
        # KS
        ks_test = kolmogorov_smirnov(labels[idx_test].cpu().detach().numpy(),
                                     ce_feats[idx_test][:, 1].cpu().detach().numpy())
        # confusion matrix
        cm = confusion_matrix(labels[idx_test].cpu().detach().numpy(),
                              ce_feats[idx_test].argmax(dim=1).cpu().detach().numpy())
        print("Batch {}".format(idx + 1),
              "loss = {:.4f}".format(loss_test),
              "AUC = {:.2f}%".format(100 * auc_test),
              "KS = {:.2f}%".format(100 * ks_test))
        print(cm)
        auc_list.append(auc_test)
        ks_list.append(ks_test)


    # Train latent_relation_model
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    for epoch in range(args.epochs):
        loss_values.append(train())

        torch.save({'gat_state_dict': model.state_dict()},
                   '../best_model_updated/best_model_{}.pth'.format(epoch))

        # Save the best latent_relation_model
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
            files = glob.glob('best_model_updated/best_model_*.pth')
            for file in files:
                epoch_nb = int(file.split('_')[-1].split('.')[0])
                if epoch_nb < best_epoch:
                    os.remove(file)
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

    # Remove all models except the best one
    files = glob.glob('best_model_updated/best_model_*.pth')
    for file in files:
        epoch_nb = int(file.split('_')[-1].split('.')[0])
        if epoch_nb != best_epoch:
            os.remove(file)

    checkpoint = torch.load(
        '../best_model_updated/best_model_{}.pth'.format(best_epoch))
    model.load_state_dict(checkpoint['gat_state_dict'])

    # Testing
    compute_test()

print("Total AUC = {}".format(auc_list))
print("Total KS = {}".format(ks_list))
print("Average AUC = {:.2f}%".format(100 * np.average(auc_list)),
      "Average KS  = {:.2f}%".format(100 * np.average(ks_list)))
