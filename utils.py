import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(label):
    class_set = set(label)
    class_dict = {c: np.identity(len(class_set))[i, :]
                        for i, c in enumerate(class_set)}
    onehot_label = np.array(list(map(class_dict.get, label)), dtype=np.int32)

    return onehot_label

def load_data(path='data/cora/', dataset='cora'):
    print('Loading {} dataset...'.format(dataset))

    idx_feature_label = np.genfromtxt('{}{}.content'.format(path, dataset), dtype=np.dtype(str))
    idx = np.array(idx_feature_label[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    feature = sp.csr_matrix(idx_feature_label[:, 1:-1], dtype=np.float32)
    label = encode_onehot(idx_feature_label[:, -1])

    edge_unordered = np.genfromtxt('{}{}.cites'.format(path, dataset), dtype=np.int32)
    edge = np.array(list(map(idx_map.get, edge_unordered.flatten())), dtype=np.int32)
    edge = edge.reshape(edge_unordered.shape)
    adj_mat = sp.coo_matrix((np.ones(edge.shape[0]), (edge[:, 0], edge[:, 1])),
                            shape=(label.shape[0], label.shape[0]), dtype=np.float32)
    adj_mat = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat) - adj_mat.multiply(adj_mat.T > adj_mat)

    feature = normalize(feature)
    adj_mat = normalize(adj_mat + sp.eye(adj_mat.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    feature = torch.FloatTensor(np.array(feature.todense()))
    label = torch.LongTensor(np.where(label)[1])
    adj_mat = sparse_mat_to_torch_sparse_tensor(adj_mat)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj_mat, feature, label, idx_train, idx_val, idx_test

def normalize(mat):
    row_sum = np.array(mat.sum(axis=1))
    row_sum_inv = np.power(row_sum, -1).flatten()
    row_sum_inv[np.isinf(row_sum_inv)] = 0.
    diag = sp.diags(row_sum_inv)
    mat_norm = diag.dot(mat)

    return mat_norm

def sparse_mat_to_torch_sparse_tensor(sparse_mat):
    sparse_mat = sparse_mat.tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack((sparse_mat.row, sparse_mat.col)).astype(np.int64))
    value = torch.from_numpy(sparse_mat.data)
    shape = torch.Size(sparse_mat.shape)

    return torch.sparse.FloatTensor(idx, value, shape)

def calculate_accuracy(output, label):
    pred = output.max(1)[1].type_as(label)
    correct = pred.eq(label).double()
    correct = correct.sum()
    out = correct / len(label)

    return out