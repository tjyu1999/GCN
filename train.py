import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import *
from utils import *
from param import *


adj_mat, feature, label, idx_train, idx_val, idx_test = load_data()

model = GCN(input_dim=feature.shape[1],
            hidden_dim=args.hidden,
            num_class=label.max().item() + 1)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    adj_mat = adj_mat.cuda()
    feature = feature.cuda()
    label = label.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

def Train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(feature, adj_mat)
    loss_train = F.nll_loss(output[idx_train], label[idx_train])
    acc_train = calculate_accuracy(output[idx_train], label[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(feature, adj_mat)

    loss_val = F.nll_loss(output[idx_val], label[idx_val])
    acc_val = calculate_accuracy(output[idx_val], label[idx_val])

    print('Epoch: {:04d}'.format(epoch + 1),
          'Training Loss: {:.4f}'.format(loss_train.item()),
          'Training Accuracy: {:.4f}'.format(acc_train.item()),
          'Validation Loss: {:.4f}'.format(loss_val.item()),
          'Validation Accuracy: {:.4f}'.format(acc_val.item()),
          'Time: {:.4f}s'.format(time.time() - t))
    print('==================================================', end='')
    print('==================================================', end='')
    print('=========================')

def Test():
    model.eval()
    output = model(feature, adj_mat)
    loss_test = F.nll_loss(output[idx_test], label[idx_test])
    acc_test = calculate_accuracy(output[idx_test], label[idx_test])

    print('Testing Result',
          'Testing Loss: {:.4f}'.format(loss_test.item()),
          'Testing Accuracy: {:.4f}'.format(acc_test.item()))

t_total = time.time()
for epoch in range(args.epoch):
    Train(epoch)
print('Optimization Finished')
print('Total Time Elapsed: {:.4f}s'.format(time.time() - t_total))
Test()