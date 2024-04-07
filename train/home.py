from LG_GNN.model.model import LG_GNN
from LG_GNN.utils import accuracy, load_data, kolmogorov_smirnov
from LG_GNN.contrastive_loss.SupervisedContrastiveLoss import SupConLoss

import torch
from torch import nn
import torch.optim as optim
import os
import glob
import time
import random
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fast_mode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=70, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden_gat', type=int, default=[128, 64], help='Number of hidden units of GAT.')
parser.add_argument('--nb_heads', type=int, default=2, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--share_weights', action='store_true', default=True,
                    help='The same matrix will be applied to the source and the target node of every edge.')
parser.add_argument('--temp', type=float, default=[0.07, 0.007], help='Temperature of contrastive learning.')
parser.add_argument('--eta', type=float, default=0.4, help='Weight of contrastive loss.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.init()

# AUC, KS
auc_list = []
ks_list = []

# Load data
# p: Home1(0.9), Home2(0.9)
# batch_size: Home1(1000), Home2(1000)
datasets = load_data(dataset_name='Home1', p=0.9, batch=1000)
for idx, batch in enumerate(datasets):
    adj = batch['adj']
    features = batch['features']
    labels = batch['labels']
    idx_train = batch['idx_train']
    idx_val = batch['idx_val']
    idx_test = batch['idx_test']

    model = LG_GNN(in_features=features.shape[1],
                   n_hidden_gnn=args.hidden_gat,
                   n_classes=2,
                   n_heads=args.nb_heads,
                   k_l=0.6,  # Home1(0.6), Home2(0.7)
                   k_g=0.8,  # Home1(0.8), Home2(0.6)
                   dropout=args.dropout,
                   num_unique_values=32,
                   embed_dim=8,
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
        output = model(features, adj)
        # loss
        loss_train_cl = criterion_cl(features[idx_train], labels[idx_train])
        loss_train_mlp = criterion(output[idx_train], labels[idx_train])
        loss_train = args.eta * loss_train_cl + (1 - args.eta) * loss_train_mlp
        # acc
        acc_train = accuracy(output[idx_train], labels[idx_train])
        # Back Propagation
        loss_train.backward()
        optimizer.step()

        if not args.fast_mode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, adj)
        # loss
        loss_val_cl = criterion_cl(features[idx_val], labels[idx_val])
        loss_val_mlp = criterion(output[idx_val], labels[idx_val])
        loss_val = args.eta * loss_val_cl + (1 - args.eta) * loss_val_mlp
        acc_val = accuracy(output[idx_val], labels[idx_val])
        # print('Epoch: {:04d}'.format(epoch + 1),
        #       'loss_train: {:.4f}'.format(loss_train.data.item()),
        #       'acc_train: {:.4f}'.format(acc_train.data.item()),
        #       'loss_val: {:.4f}'.format(loss_val.data.item()),
        #       'acc_val: {:.4f}'.format(acc_val.data.item()))
        return loss_val.data.item()


    def compute_test():
        model.eval()
        output = model(features, adj)
        # loss
        loss_test_cl = criterion_cl(features[idx_test], labels[idx_test])
        loss_test_mlp = criterion(output[idx_test], labels[idx_test])
        loss_test = args.eta * loss_test_cl + (1 - args.eta) * loss_test_mlp

        # AUC
        auc_test = roc_auc_score(labels[idx_test].cpu().detach().numpy(),
                                 output[idx_test][:, 1].cpu().detach().numpy())
        # KS
        ks_test = kolmogorov_smirnov(labels[idx_test].cpu().detach().numpy(),
                                     output[idx_test][:, 1].cpu().detach().numpy())
        print("Batch {}".format(idx + 1),
              "loss = {:.4f}".format(loss_test),
              "AUC = {:.2f}%".format(100 * auc_test),
              "KS = {:.2f}%".format(100 * ks_test))
        auc_list.append(auc_test)
        ks_list.append(ks_test)


    # Train model
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    for epoch in range(args.epochs):
        loss_values.append(train())

        # Save the best model
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
            # Clean up
            files = glob.glob('best_model_updated/best_model_*.pth')
            for file in files:
                os.remove(file)
            # Save
            torch.save({'state_dict': model.state_dict()},
                       '../best_model_updated/best_model.pth')

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
        '../best_model_updated/best_model.pth'.format(best_epoch))
    model.load_state_dict(checkpoint['state_dict'])

    # Testing
    compute_test()

print()
print("Average AUC = {:.2f}%".format(100 * np.average(auc_list)))
print("Average KS = {:.2f}%".format(100 * np.average(ks_list)))
