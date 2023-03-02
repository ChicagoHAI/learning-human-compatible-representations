import numpy as np
import matplotlib.pyplot as plt

### synthetic distorted human intuition / metric

def trans(x, w):
    return np.diag(w).dot(x.T).T

def plot(x, xt, figsize=(13, 6), xlim=[-0.2, 1], ylim=[-0.2, 1]):
    fig, ax = plt.subplots(1,2,figsize=figsize)
    ax[0].set_title('feature'); ax[1].set_title('human')
    ax[0].scatter(x[:,0][y==0], x[:,1][y==0])
    ax[0].scatter(x[:,0][y==1], x[:,1][y==1])
    ax[1].scatter(xt[:,0][y==0], xt[:,1][y==0])
    ax[1].scatter(xt[:,0][y==1], xt[:,1][y==1])
    ax[0].set_xlim(xlim); ax[1].set_xlim(xlim)
    ax[0].set_ylim(ylim); ax[1].set_ylim(ylim)
    return fig, ax

from sklearn.metrics.pairwise import euclidean_distances as euc_dist
def ord_dist(a, b, order=2):
    order = np.array(order) if type(order) != int else order
    root = 2 if type(order) != int and len(order) > 1 else order
    diff = a[:,np.newaxis].repeat(len(b),1) - b
    return (np.abs(diff)**order).sum(-1)**(1/root)


### putting near in leftmost

def get_nn_tup(dist, y_test, y_train):
    mask_test = np.tile(y_test, (len(y_train), 1)).T
    mask_train = np.tile(y_train, (len(y_test), 1))
    mask_in = mask_test == mask_train
    mask_out = mask_test != mask_train
    apply_mask = lambda x, m: x + (-(m - 1) * x.max())
    in1nn = np.argmin(apply_mask(dist, mask_in), 1)
    out1nn = np.argmin(apply_mask(dist, mask_out), 1)
    nn_tup = np.vstack([np.arange(len(y_test)), in1nn, out1nn])
    return nn_tup.T

def eval_nn_tup(dist, nn_tup, y_test, y_train):
    dst = dist.take(nn_tup[:,0], 0)
    dap = np.take_along_axis(dst, nn_tup[:,1].reshape(-1,1), 1).ravel()
    dan = np.take_along_axis(dst, nn_tup[:,2].reshape(-1,1), 1).ravel()
    pp = y_train.take(nn_tup[:,1]).ravel()
    pn = y_train.take(nn_tup[:,2]).ravel()
    pa = np.zeros((len(nn_tup)))
    pa[dap < dan] = pp[dap < dan]
    pa[dap >= dan] = pn[dap >= dan]
    y_true = y_test.take(nn_tup[:,0])
    return pa == y_true


### putting support from each class in order of unique labels

def get_ds(dist, y_test, y_train):
    mask_train = np.tile(y_train, (len(y_test), 1))
    apply_mask = lambda x, m: x + (-(m - 1) * x.max())
    ds = np.arange(len(y_test)).reshape(-1, 1)
    for label in np.sort(np.unique(y_train)):
        mask_in = label == mask_train
        in1nn = np.argmin(apply_mask(dist, mask_in), 1)
        ds = np.hstack([ds, in1nn.reshape(-1, 1)])
    return ds

def eval_ds(dist, ds, y_test, y_train):
    dst = dist.take(ds[:,0], 0)
    dnn = np.vstack([np.take_along_axis(
        dst, ds[:, c].reshape(-1,1), 1).ravel() for c in np.arange(1, ds.shape[1])])
    y_pred = np.unique(y_train).take(dnn.argmin(0))
    y_true = y_test.take(ds[:,0])
    return y_pred == y_true

def get_ds_choice(dist, ds):
    dst = dist.take(ds[:,0], 0)
    dnn = np.vstack([np.take_along_axis(
        dst, ds[:, c].reshape(-1,1), 1).ravel() for c in np.arange(1, ds.shape[1])])
    return dnn.argmin(0)

def get_ds_chosen(choice, ds):
    chosen = np.take_along_axis(ds[:,1:], choice.reshape(-1, 1), 1).ravel()
    return chosen

def eval_ds_choice(choice, ds, y_test, y_train):
    chosen = np.take_along_axis(ds[:,1:], choice.reshape(-1, 1), 1).ravel()
    y_pred = y_train.take(chosen)
    y_true = y_test.take(ds[:,0])
    return y_pred == y_true
