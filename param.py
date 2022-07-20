import argparse
import numpy as np
import torch


parser = argparse.ArgumentParser()

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='validate during training pass')
parser.add_argument('--seed', type=int, default=20,
                    help='random seed')
parser.add_argument('--epoch', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight decay')
parser.add_argument('--hidden', type=int, default=16,
                    help='number of hidden units')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout rate')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)