import argparse
import os
import random
from time import time

import torch
import torch.backends.cudnn as cudnn

import onlinehd
from dataloader import load_isolet
from utils import save_model_linear


# simple OnlineHD training
def main(args):
    print('Loading...')
    x, x_test, y, y_test = load_isolet(args.data, normalize=False)
    classes = y.unique().size(0)
    features = x.size(1)
    model = onlinehd.OnlineHD(classes, features, linear_encoder=True)

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        x_test = x_test.cuda()
        y_test = y_test.cuda()
        model = model.to('cuda')
        print('Using GPU!')

    print('Encoding...')
    h = model.encode(x)
    h_test = model.encode(x_test)

    print('Training...')
    t = time()
    model = model.fit(h, y, encoded=True, bootstrap=args.bootstrap, lr=args.lr, epochs=args.epochs, one_pass_fit=args.one_pass_fit)
    t = time() - t

    print('Validating...')
    yhat = model(h, encoded=True)
    yhat_test = model(h_test, encoded=True)
    acc = (y == yhat).float().mean()
    acc_test = (y_test == yhat_test).float().mean()
    print(f'{acc = :6f}')
    print(f'{acc_test = :6f}')
    print(f'{t = :6f}')

    # Save result
    save_model_linear(model, os.path.join(args.results, 'model.pth'))
    with open(os.path.join(args.results, 'results.txt'), 'a') as wf:
        wf.write('acc = {}\nacc_test = {}\nt = {}'.format(acc, acc_test, t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.035, type=float, metavar='L')
    parser.add_argument('--epochs', default=40, type=int, metavar='E')
    parser.add_argument('--dimension', default=4000, type=int, metavar='D')
    parser.add_argument('--bootstrap', default=1.0, type=float, metavar='B')
    parser.add_argument('--one_pass_fit', default=True, type=bool, metavar='O')
    parser.add_argument('--data', default='./data', type=str)
    parser.add_argument('--results', default='./results', type=str)
    parser.add_argument('--seed', default=103, type=int)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # Save args
    os.makedirs(args.results, exist_ok=True)
    with open(os.path.join(args.results, 'results.txt'), 'w') as wf:
        wf.write(str(args) + '\n')

    main(args)
