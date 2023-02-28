from __future__ import division, print_function
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('agg')
from scipy.optimize import linear_sum_assignment as linear_assignment
import random
import os
import argparse

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

from sklearn import metrics
import time

# -------------------------------
# Evaluation Criteria
# -------------------------------
def evaluate_clustering(y_true, y_pred):

    start = time.time()
    print('Computing metrics...')
    if len(set(y_pred)) < 1000:
        acc = cluster_acc(y_true.astype(int), y_pred.astype(int))
    else:
        acc = None

    nmi = nmi_score(y_true, y_pred)
    ari = ari_score(y_true, y_pred)
    pur = purity_score(y_true, y_pred)
    print(f'Finished computing metrics {time.time() - start}...')

    return acc, nmi, ari, pur


def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    # __import__("ipdb").set_trace()
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

# -------------------------------
# Mixed Eval Function
# -------------------------------
def mixed_eval(targets, preds, mask):

    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    :param targets: All ground truth labels
    :param preds: All predictions
    :param mask: Mask defining two subsets
    :return:
    """

    mask = mask.astype(bool)

    # Labelled examples
    if mask.sum() == 0:  # All examples come from unlabelled classes

        unlabelled_acc, unlabelled_nmi, unlabelled_ari = cluster_acc(targets.astype(int), preds.astype(int)), \
                                                         nmi_score(targets, preds), \
                                                         ari_score(targets, preds)

        print('Unlabelled Classes Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'
              .format(unlabelled_acc, unlabelled_nmi, unlabelled_ari))

        # Also return ratio between labelled and unlabelled examples
        return (unlabelled_acc, unlabelled_nmi, unlabelled_ari), mask.mean()

    else:

        labelled_acc, labelled_nmi, labelled_ari = cluster_acc(targets.astype(int)[mask],
                                                               preds.astype(int)[mask]), \
                                                   nmi_score(targets[mask], preds[mask]), \
                                                   ari_score(targets[mask], preds[mask])

        unlabelled_acc, unlabelled_nmi, unlabelled_ari = cluster_acc(targets.astype(int)[~mask],
                                                                     preds.astype(int)[~mask]), \
                                                         nmi_score(targets[~mask], preds[~mask]), \
                                                         ari_score(targets[~mask], preds[~mask])

        # Also return ratio between labelled and unlabelled examples
        return (labelled_acc, labelled_nmi, labelled_ari), (
            unlabelled_acc, unlabelled_nmi, unlabelled_ari), mask.mean()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class BCE(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()


def PairEnum(x,mask=None):

    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))

    if mask is not None:

        xmask = mask.view(-1, 1).repeat(1, x.size(1))
        #dim 0: #sample, dim 1:#feature 
        x1 = x1[xmask].view(-1, x.size(1))
        x2 = x2[xmask].view(-1, x.size(1))

    return x1, x2


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import torch
from torch import mvlgamma
from torch import lgamma
import numpy as np


class Priors:
    '''
    A prior that will hold the priors for all the parameters.
    '''
    def __init__(self, args, K, codes_dim, counts=10, prior_sigma_scale=None):
        self.name = "prior_class"
        self.pi_prior_type = args.pi_prior # uniform
        if args.pi_prior:
            self.pi_prior = Dirichlet_prior(K, args.pi_prior, counts)
        else:
            self.pi_prior = None
        self.mus_covs_prior = NIW_prior(args, prior_sigma_scale)

        self.name = self.mus_covs_prior.name
        self.pi_counts = args.prior_dir_counts # 0.1

    def update_pi_prior(self, K_new, counts=10, pi_prior=None):
        # pi_prior = None- keep the same pi_prior type
        if self.pi_prior:
            if pi_prior:
                self.pi_prioir = Dirichlet_prior(K_new, pi_prior, counts)
            self.pi_prior = Dirichlet_prior(K_new, self.pi_prior_type, counts)

    def comp_post_counts(self, counts):
        if self.pi_prior:
            return self.pi_prior.comp_post_counts(counts)
        else:
            return counts

    def comp_post_pi(self, pi):
        if self.pi_prior:
            return self.pi_prior.comp_post_pi(pi, self.pi_counts)
        else:
            return pi

    def get_sum_counts(self):
        return self.pi_prior.get_sum_counts()

    def init_priors(self, codes):
        return self.mus_covs_prior.init_priors(codes)

    def compute_params_post(self, codes_k, mu_k):
        return self.mus_covs_prior.compute_params_post(codes_k, mu_k)

    def compute_post_mus(self, N_ks, data_mus):
        return self.mus_covs_prior.compute_post_mus(N_ks, data_mus)

    def compute_post_cov(self, N_k, mu_k, data_cov_k):
        return self.mus_covs_prior.compute_post_cov(N_k, mu_k, data_cov_k)

    def log_marginal_likelihood(self, codes_k, mu_k):
        return self.mus_covs_prior.log_marginal_likelihood(codes_k, mu_k)


class Dirichlet_prior:
    def __init__(self, K, pi_prior="uniform", counts=10):
        self.name = "Dirichlet_dist"
        self.K = K
        self.counts = counts
        if pi_prior == "uniform":
            self.p_counts = torch.ones(K) * counts
            self.pi = self.p_counts / float(K * counts)

    def comp_post_counts(self, counts=None):
        if counts is None:
            counts = self.counts
        return counts + self.p_counts

    def comp_post_pi(self, pi, counts=None):
        if counts is None:
            # counts = 0.001
            counts = 0.1
        return (pi + counts) / (pi + counts).sum()

    def get_sum_counts(self):
        return self.K * self.counts


class NIW_prior:
    """A class used to store niw parameters and compute posteriors.
    Used as a class in case we will want to update these parameters.
    """

    def __init__(self, args, prior_sigma_scale=None):
        self.name = "NIW"
        self.prior_mu_0_choice = args.prior_mu_0 # data_mean
        self.prior_sigma_choice = args.prior_sigma_choice # isotropic
        self.prior_sigma_scale = prior_sigma_scale or args.prior_sigma_scale #  .005
        self.niw_kappa = args.prior_kappa # 0.0001
        self.niw_nu = args.prior_nu # at least feat_dim + 1
        

    def init_priors(self, codes):
        if self.prior_mu_0_choice == "data_mean":
            self.niw_m = codes.mean(axis=0)
        if self.prior_sigma_choice == "isotropic":
            self.niw_psi = (torch.eye(codes.shape[1]) * self.prior_sigma_scale).double()
        elif self.prior_sigma_choice == "data_std":
            self.niw_psi = (torch.diag(codes.std(axis=0)) * self.prior_sigma_scale).double()
        else:
            raise NotImplementedError()
        return self.niw_m, self.niw_psi

    def compute_params_post(self, codes_k, mu_k):
        # This is in HARD assignment.
        N_k = len(codes_k)
        sum_k = codes_k.sum(axis=0)
        kappa_star = self.niw_kappa + N_k
        nu_star = self.niw_nu + N_k
        mu_0_star = (self.niw_m * self.niw_kappa + sum_k) / kappa_star
        codes_minus_mu = codes_k - mu_k
        S = codes_minus_mu.T @ codes_minus_mu
        psi_star = (
            self.niw_psi
            + S
            + (self.niw_kappa * N_k / kappa_star)
            * (mu_k - self.niw_m).unsqueeze(1)
            @ (mu_k - self.niw_m).unsqueeze(0)
        )
        return kappa_star, nu_star, mu_0_star, psi_star

    def compute_post_mus(self, N_ks, data_mus):
        # N_k is the number of points in cluster K for hard assignment, and the sum of all responses to the K-th cluster for soft assignment
        return ((N_ks.reshape(-1, 1) * data_mus) + (self.niw_kappa * self.niw_m)) / (
            N_ks.reshape(-1, 1) + self.niw_kappa
        )

    def compute_post_cov(self, N_k, mu_k, data_cov_k):
        # If it is hard assignments: N_k is the number of points assigned to cluster K, x_mean is their average
        # If it is soft assignments: N_k is the r_k, the sum of responses to the k-th cluster, x_mean is the data average (all the data)
        D = len(mu_k)
        if N_k > 0:
            return (
                self.niw_psi
                + data_cov_k * N_k  # unnormalize
                + (
                    ((self.niw_kappa * N_k) / (self.niw_kappa + N_k))
                    * ((mu_k - self.niw_m).unsqueeze(1) * (mu_k - self.niw_m).unsqueeze(0))
                )
            ) / (self.niw_nu + N_k + D + 2)
        else:
            return self.niw_psi

    def log_marginal_likelihood(self, codes_k, mu_k):
        kappa_star, nu_star, mu_0_star, psi_star = self.compute_params_post(
            codes_k, mu_k
        )
        (N_k, D) = codes_k.shape
        return (
            -(N_k * D / 2.0) * np.log(np.pi)
            + mvlgamma(torch.tensor(nu_star / 2.0), D)
            - mvlgamma(torch.tensor(self.niw_nu) / 2.0, D)
            + (self.niw_nu / 2.0) * torch.logdet(self.niw_psi)
            - (nu_star / 2.0) * torch.logdet(psi_star)
            + (D / 2.0) * (np.log(self.niw_kappa) - np.log(kappa_star))
        )

