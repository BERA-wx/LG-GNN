import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_curve
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import cosine_similarity

'''utils'''

seed = 42


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def kolmogorov_smirnov(labels, pred):
    fpr, tpr, thresholds = roc_curve(labels, pred)
    ks_value = max(tpr - fpr)
    return ks_value


'''load data'''


class LoanDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # specified index sample
        features = self.data.iloc[index, :-1].values.astype(np.float32)
        labels = self.data.iloc[index, -1].astype(np.float32)
        return features, labels


def load_data(dataset_name, p=0.9, batch=1000):
    """
    get features_dn and labels_dn：
    - features_dn: float 32
    - labels_dn: float 32
    """
    # data
    dataset_paths = {
        'Lending1': '../datasets/preprocessed/Lending1.csv',
        'Lending2': '../datasets/preprocessed/Lending2.csv',
        'Lending3': '../datasets/preprocessed/Lending3.csv',
        'Home1': '../datasets/preprocessed/home1.csv',
        'Home2': '../datasets/preprocessed/home2.csv',
        'PPD1': '../datasets/preprocessed/ppd1.csv',
        'PPD2': '../datasets/preprocessed/ppd2.csv',
    }

    if dataset_name in dataset_paths:
        path = dataset_paths[dataset_name]
    else:
        path = None

    feats_labels = pd.read_csv(path)
    dataset = LoanDataset(feats_labels)

    # batch
    batch_size = batch

    def calculate_adjacency_matrix(input_batch):
        # features & labels
        feats = csr_matrix(np.stack([item[0] for item in input_batch], axis=0), dtype=np.float32)
        lbs = encode_onehot(np.stack([item[1] for item in input_batch], axis=0))

        # calculate the adjacency matrix
        similarity_matrix = cosine_similarity(feats)
        threshold = p
        adj = csr_matrix(similarity_matrix > threshold, dtype=int)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # normalization
        adj = normalize(adj)
        adj = normalize(adj + csr_matrix(np.eye(adj.shape[0]), dtype=np.float32))

        # format transformation
        feats = torch.FloatTensor(np.array(feats.todense()))
        lbs = torch.LongTensor(np.where(lbs)[1])
        adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense().unsqueeze(-1)

        return adj, feats, lbs

    # custom data loader
    def collate_fn(input_batch):
        # data
        adj, feats, lbs = calculate_adjacency_matrix(input_batch)

        # train, val, test split
        num_samples = feats.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        # train:val:test = 80%:10%:10%
        index_train = torch.LongTensor(indices[: int(0.8 * num_samples)])
        index_val = torch.LongTensor(indices[int(0.8 * num_samples): int(0.9 * num_samples)])
        index_test = torch.LongTensor(indices[int(0.9 * num_samples):])

        return adj, feats, lbs, index_train, index_val, index_test

    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=collate_fn)

    processed_batches = []
    for idx, batch in enumerate(dataloader):
        adj, features, labels, idx_train, idx_val, idx_test = batch
        # add to list
        processed_data = {
            'adj': adj,
            'features': features,
            'labels': labels,
            'idx_train': idx_train,
            'idx_val': idx_val,
            'idx_test': idx_test
        }
        processed_batches.append(processed_data)
    return processed_batches
