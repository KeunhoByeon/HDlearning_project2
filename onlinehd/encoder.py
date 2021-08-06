import math

import numpy as np
import torch
from tqdm import tqdm


class Encoder(object):
    '''
    The nonlinear encoder class maps data nonlinearly to high dimensional space.
    To do this task, it uses two randomly generated tensors:

    :math:`B`. The `(dim, features)` sized random basis hypervectors, drawn
    from a standard normal distribution
    :math:`b`. An additional `(dim,)` sized base, drawn from a uniform
    distribution between :math:`[0, 2\pi]`.

    The hypervector :math:`H \in \mathbb{R}^D` of :math:`X \in \mathbb{R}^f`
    is:

    .. math:: H_i = \cos(X \cdot B_i + b_i) \sin(X \cdot B_i)

    Args:
        features (int, > 0): Dimensionality of original data.

        dim (int, > 0): Target dimension for output data.
    '''

    def __init__(self, features: int, dim: int = 4000):
        self.dim = dim
        self.features = features
        self.basis = torch.randn(self.dim, self.features)
        self.base = torch.empty(self.dim).uniform_(0.0, 2 * math.pi)

    def __call__(self, x: torch.Tensor):
        '''
        Encodes each data point in `x` to high dimensional space.
        The encoded representation of the `(n?, features)` samples described
        in :math:`x`, is the `(n?, dim)` matrix :math:`H`:

        .. math:: H_{ij} = \cos(x_i \cdot B_j + b_j) \sin(x_i \cdot B_j)

        Note:
            This encoder is very sensitive to data preprocessing. Try
            making input have unit norm (normalizing) or standarizing each
            feature to have mean=0 and std=1/sqrt(features) (scaling).

        Args:
            x (:class:`torch.Tensor`): The original data points to encode. Must
                have size `(n?, features)`.

        Returns:
            :class:`torch.Tensor`: The high dimensional representation of each
            of the `n?` data points in x, which respects the equation given
            above. It has size `(n?, dim)`.
        '''

        n = x.size(0)
        bsize = math.ceil(0.01 * n)
        h = torch.empty(n, self.dim, device=x.device, dtype=x.dtype)
        temp = torch.empty(bsize, self.dim, device=x.device, dtype=x.dtype)

        # we need batches to remove memory usage
        for i in range(0, n, bsize):
            torch.matmul(x[i:i + bsize], self.basis.T, out=temp)
            torch.add(temp, self.base, out=h[i:i + bsize])
            h[i:i + bsize].cos_().mul_(temp.sin_())
        return h

    def to(self, *args):
        '''
        Moves data to the device specified, e.g. cuda, cpu or changes
        dtype of the data representation, e.g. half or double.
        Because the internal data is saved as torch.tensor, the parameter
        can be anything that torch accepts. The change is done in-place.

        Args:
            device (str or :class:`torch.torch.device`) Device to move data.

        Returns:
            :class:`Encoder`: self
        '''

        self.basis = self.basis.to(*args)
        self.base = self.base.to(*args)
        return self


class LinearEncoder(object):
    def __init__(self, features: int, dim: int = 4000, m=1000):
        self.dim = dim
        self.features = features
        self.m = m
        self.basis = torch.randint(0, 2, size=(self.dim, self.features)) * 2 - 1
        self.level = np.random.randint(2, size=self.dim) * 2 - 1

        self.levels = []
        temp = np.ones(self.dim)
        for i in range(1, self.m + 1):
            # flip = int(self.dim / i)
            flip = int(self.dim / m) * i
            temp[:flip] = -1
            temp[flip:] = 1
            self.levels.append(np.array(self.level * temp))
        self.levels = torch.from_numpy(np.array(self.levels))

    def __call__(self, x: torch.Tensor):
        n = x.size(0)
        bsize = math.ceil(0.05 * n)
        h = torch.empty(n, self.dim, device=x.device, dtype=x.dtype)

        # we need batches to remove memory usage
        temp = torch.empty(bsize, self.features, self.dim)
        for i in tqdm(range(0, n, bsize), leave=False):
            try:
                l = ((x[i:i + bsize].cpu().detach().numpy() + 1) / 2 * (self.m - 1)).astype(int)
                temp = self.levels[l] * self.basis.T
                temp = temp.view(-1, self.features, self.dim)
                h[i:i + bsize] = torch.sum(temp, dim=1)
            except Exception as e:
                print(i, '/', n)
                print('l', l.shape)
                print('self.levels', self.levels.shape)
                print('self.basis.T', self.basis.T.shape)
                print(temp.shape)
                print(self.levels[l] * self.basis.T)
                raise e

        return h

    def to(self, *args):
        self.basis = self.basis.to(*args)
        self.levels = self.levels.to(*args)
        return self


if __name__ == '__main__':
    from dataloader import load_isolet

    e = LinearEncoder(617)
    x, x_test, _, _ = load_isolet(data_dir='../data', normalize=False)
    if torch.cuda.is_available():
        x= x.cuda()
        x_test = x_test.cuda()
        e.to(x.device)
    # print(e(x).size())
    print(e(x_test).size())
