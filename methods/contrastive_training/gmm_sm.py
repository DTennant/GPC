import argparse
import os

from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import torch
from torch.optim import SGD, lr_scheduler
import torchvision as tv
from project_utils.cluster_utils import mixed_eval, AverageMeter
from models import vision_transformer as vits
from models import wrn


from project_utils.general_utils import init_experiment, get_mean_lr, str2bool, get_dino_head_weights, accuracy

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from tqdm import tqdm

from torch.nn import functional as F
from torch import nn
from loguru import logger

from project_utils.cluster_and_log_utils import log_accs_from_preds
from project_utils.cluster_utils import cluster_acc, np, linear_assignment
# from methods.clustering.k_means import kmeans_faiss
from methods.clustering.faster_mix_k_means_pytorch import K_Means, pairwise_distance
from config import exp_root, dino_pretrain_path, mae_inat_pretrain_path
from data.get_datasets import customize_datasets
from project_utils.loss_utils import info_nce_logits, SupConLoss, ContrastiveLearningViewGenerator, StrongWeakView
from project_utils.loss_utils import prototypical_logits, proto_for_supcon_logits


from project_utils.split_and_merge_ops import init_mus_and_covs, init_mus_and_covs_sub, compute_split_log_ll
from project_utils.split_and_merge_ops import compute_data_covs_hard_assignment, log_Hastings_ratio_split, log_Hastings_ratio_merge
from project_utils.cluster_utils import Priors

# TODO: Debug
import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", )


def get_sub_assign_with_one_cluster(feat, labels, k, prior):
    counts = []
    class_indices = labels == k
    class_sub_feat = feat[class_indices]
    
    if len(class_sub_feat) <= 2:
        c = torch.tensor([0, len(class_sub_feat)])
        class_sub_assign = torch.ones(len(class_sub_feat), dtype=torch.long)
        mu_subs = torch.mean(class_sub_feat, dim=0, keepdim=True)
        mu_subs = torch.cat([torch.zeros_like(mu_subs), mu_subs], dim=0)
        # NOTE: empty sub clusters
    else:
        km = K_Means(k=2, init='k-means++', random_state=1, n_jobs=None, pairwise_batch_size=128)
        km.fit(class_sub_feat)
        class_sub_assign = km.labels_.cpu()
        mu_subs = km.cluster_centers_
        _, c = torch.unique(class_sub_assign, return_counts=True)
    counts.extend(c.cpu().numpy().tolist())

    data_covs_sub = compute_data_covs_hard_assignment(class_sub_assign, class_sub_feat, 2, mu_subs.cpu(), prior)
    
    # update prior
    mu_subs = prior.compute_post_mus(torch.tensor(counts), mu_subs.cpu())
    covs_sub = []
    for k in range(2):
        covs_sub_k = prior.compute_post_cov(counts[k], class_sub_feat[class_sub_assign == k].mean(axis=0), data_covs_sub[k])
        covs_sub.append(covs_sub_k)
    covs_sub = torch.stack(covs_sub)
    
    pi_sub = torch.tensor(counts) / float(len(class_sub_feat))
    return mu_subs, covs_sub, pi_sub, class_sub_assign

def get_sub_cluster_with_sskmeans(u_feat, l_feat, l_targets, labels, prior, args,):
    # NOTE: reads cluster assignments from sskmeans, perform clustering within each cluster
    sub_clusters = []
    # only the unlabelled data will be splited or merged
    l_labels = labels[:len(l_targets)].unique().cpu()
    all_labels = labels[len(l_targets):].unique().cpu()
    
    for class_label in all_labels.cpu().numpy().tolist():
        mu_sub, cov_sub, pi_sub, class_sub_assign = get_sub_assign_with_one_cluster(u_feat, labels[len(l_targets):], class_label, prior)
        sub_clusters.append([class_label, (mu_sub, cov_sub, pi_sub, class_sub_assign)])
    return sub_clusters
    

def split_rule(feats, sub_assignment, prior, mu, mu_subs):
    # NOTE: deal with empty clusters first, pi_sub is [0, 1], no split
    """
    feats: NxD, subset of features
    sub_assignment: N, 0 and 1 assignments
    mu: 1xD, cluster center
    mu_subs: 2xD, sub cluster centers
    return [k, bool], split the k-th cluster or not
    """
    if len(feats) <= 5:
        # small clusters will not be splited
        return False
    
    if len(feats[sub_assignment == 0]) <= 5 or len(feats[sub_assignment == 1]) <= 5:
        return False
    
    log_ll_k = prior.log_marginal_likelihood(feats, mu)
    log_ll_k1 = prior.log_marginal_likelihood(feats[sub_assignment == 0], mu_subs[0])
    log_ll_k2 = prior.log_marginal_likelihood(feats[sub_assignment == 1], mu_subs[1])
    N_k_1 = len(feats[sub_assignment == 0])
    N_k_2 = len(feats[sub_assignment == 1])
    
    return log_Hastings_ratio_split(1.0, N_k_1, N_k_2, log_ll_k1, log_ll_k2, log_ll_k, split_prob=0.1)
    

def merge_rule(mu1, cov1, pi1, mu2, cov2, pi2, feat1, feat2, prior=None):
    all_feat = torch.cat([feat1, feat2], dim=0)
    N_k_1 = feat1.shape[0]
    N_k_2 = feat2.shape[0]
    N_k = feat1.shape[0] + feat2.shape[0]
    
    if N_k > 0:
        mus_mean = (N_k_1 / N_k) * mu1 + (N_k_2 / N_k) * mu2
    else:
        # in case both are empty clusters
        mus_mean = torch.mean(torch.stack([mu1, mu2]), axis=0)
    if prior is None:
        raise NotImplementedError
    else:
        log_ll_k = prior.log_marginal_likelihood(all_feat, mus_mean)
        log_ll_k_1 = prior.log_marginal_likelihood(feat1, mu1)
        log_ll_k_2 = prior.log_marginal_likelihood(feat2, mu2)
        
    return log_Hastings_ratio_merge(1.0, N_k_1, N_k_2, log_ll_k_1, log_ll_k_2, log_ll_k, merge_prob=0.1)
    
def split_and_merge_op(u_feat, l_feat, l_targets, args, index=0, stage=0):
    class_num = args.num_cluster[0]
    results = {
        'centroids': [],
        'density': [],
        'im2cluster': [],
    }

    # km = K_Means(k=class_num, init='k-means++', random_state=1, n_jobs=None, pairwise_batch_size=128, use_gpu=True)
    km = K_Means(k=class_num, init='k-means++', random_state=1, n_jobs=None, pairwise_batch_size=128)
    cat_feat = torch.cat((l_feat, u_feat), dim=0)

    km.fit_mix(u_feat, l_feat, l_targets, )
    
    
    centroids = km.cluster_centers_
    labels = km.labels_.cpu()
    pred = labels

    cat_feat, u_feat, l_feat, l_targets = cat_feat.cpu(), u_feat.cpu(), l_feat.cpu(), l_targets.cpu()

    prior = Priors(args, class_num, cat_feat.shape[1], )
    prior.init_priors(cat_feat)

    _, counts = torch.unique(labels, return_counts=True)
    counts = counts.cpu()
    pi = counts / float(len(cat_feat))
    data_covs = compute_data_covs_hard_assignment(labels, cat_feat, class_num, centroids.cpu(), prior)

    # NOTE: the following is to update the mu and cov using a prior. Can be disabled.
    mus = prior.compute_post_mus(counts, centroids.cpu())
    covs = []
    for k in range(len(centroids)):
        feat_k = cat_feat[labels == k]
        cov_k = prior.compute_post_cov(counts[k], feat_k.mean(axis=0), data_covs[k])  # len(codes_k) == counts[k]? yes
        covs.append(cov_k)
    covs = torch.stack(covs)
    
    # NOTE: now we have mus, covs, pi, labels for the global GMM
    sub_clusters = get_sub_cluster_with_sskmeans(u_feat, l_feat, l_targets, labels, prior, args)
    # NOTE: now we have sub_mus, sub_assignments, we can compute split rules now
    labelled_clusters = labels[:len(l_targets)].unique()

    split_decisions = []
    for class_label, items in sub_clusters:
        if class_label in labelled_clusters:
            # NOTE: labelled clusters are not considered
            continue
        class_indices = labels == class_label
        mu_subs, cov_subs, pi_subs, sub_assign = items
        split_decision = split_rule(cat_feat[class_indices], sub_assign, prior, mus[class_label], mu_subs)
        split_decisions.append([class_label, split_decision])
    
    remain_for_merge = np.array([class_l for class_l, split_d in split_decisions if not split_d])
    remain_mus = centroids[remain_for_merge].cpu()
    remain_covs = covs[remain_for_merge]
    remain_pi = pi[remain_for_merge]

    # import ipdb; ipdb.set_trace()
    merge_decisions = []
    # each cluster will only be tested for merging with the top-1 nearest cluster 
    mu_nn = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(remain_mus.cpu())

    for remain_idx, class_label in enumerate(remain_for_merge):
        nn = mu_nn.kneighbors(centroids[class_label].reshape(1, -1).cpu(), return_distance=False)[0][1:]
        nn = nn.item()
        merge_decision = merge_rule(remain_mus[remain_idx], remain_covs[remain_idx], remain_pi[remain_idx], 
                                    remain_mus[nn], remain_covs[nn], remain_pi[nn], 
                                    cat_feat[labels == class_label], cat_feat[labels == remain_for_merge[nn]], 
                                    prior)
        merge_decisions.append([class_label, merge_decision, nn])

    # NOTE: now we have split_decisions and merge_decisions, we can update the results
    new_centroids = None
    not_updated_idx = labelled_clusters.cpu().numpy().tolist()
    not_updated_idx+= [idx for idx, split_d in split_decisions if not split_d]
    not_updated_idx+= [idx for idx, merge_d, nn in merge_decisions if not merge_d]
    not_updated_idx = list(set(not_updated_idx))
    
    new_centroids = centroids[not_updated_idx].cpu()

    # perform split
    for class_label, split_d in split_decisions:
        if split_d:
            mu_subs = sub_clusters[class_label][-1][0]
            new_centroids = torch.cat((new_centroids, mu_subs))
            
    # perform merge
    for class_label, merge_d, nn in merge_decisions:
        if merge_d:
            nn_class_label = remain_for_merge[nn]
            mean_mu = (centroids[class_label] + centroids[nn_class_label]) / 2
            new_centroids = torch.cat((new_centroids, mean_mu.reshape(1, -1).cpu()))
            
    centroids = new_centroids 

    dist = pairwise_distance(cat_feat.cpu(), centroids.cpu())
    _, pred = torch.min(dist, dim=1)
    pred[:len(l_targets)] = l_targets

    # update densities
    dist2center = torch.sum((cat_feat.cpu() - centroids[pred].cpu()) ** 2, dim=1).sqrt()
    
    density = torch.zeros(len(centroids))
    for center_id in pred.unique():
        if (pred == center_id).nonzero().shape[0] > 1:
            item = dist2center[pred == center_id]
            density[center_id.item()] = item.mean() / np.log(len(item) + 10)

    dmax = density.max()
    for center_id in pred.unique():
        if (pred == center_id).nonzero().shape[0] <= 1:
            density[center_id.item()] = dmax
            
    density = density.cpu().numpy()

    density = density.clip(np.percentile(density, 10), np.percentile(density, 90)) #clamp extreme values for stability
    density = args.temperature * density / density.mean()  #scale the mean to temperature 
    density = torch.from_numpy(density).cuda()

    centroids = F.normalize(centroids, p=2, dim=1).cuda()
    results['centroids'].append(centroids)
    results['density'].append(density)
    results['im2cluster'].append(pred)    
    
    return results, None


def extract_features(model, loader, return_batch=False, no_norm=False, args=None):
    if args is not None:
        args.logger.info('Computing all features')
    model.eval()
    features, all_labels, uq_idxs = [], [], []
    for i, (images, labels, idx, is_label) in enumerate(tqdm(loader)):
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            feat = model(images, ) 
            if not no_norm:
                feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
            features.append(feat.cpu())
            all_labels.append(labels.cpu())
            uq_idxs.append(idx)

    if return_batch:
        return features, all_labels, uq_idxs 

    return torch.cat(features, dim=0).cpu(), torch.cat(all_labels, dim=0), torch.cat(uq_idxs, dim=0)

def run_kmeans(u_feat, l_feat, l_targets, args, index=0):
    # NOTE: index determine whether or not to over cluster
    # 0 is no
    class_num = args.num_cluster[index]
    results = {
        'centroids': [],
        'density': [],
        'im2cluster': [],
    }

    km = K_Means(k=class_num, init='k-means++', random_state=1, n_jobs=None, pairwise_batch_size=128, )
    
    cat_feat = torch.cat((l_feat, u_feat), dim=0)
    if index == 0:
        km.fit_mix(u_feat, l_feat, l_targets, )
    else:
        km.fit(cat_feat)
    
    centroids = km.cluster_centers_
    pred = km.labels_
    
    dist2center = torch.sum((cat_feat - centroids[pred]) ** 2, dim=1).sqrt()
    
    density = torch.zeros(class_num)
    for center_id in pred.unique():
        if (pred == center_id).nonzero().shape[0] > 1:
            item = dist2center[pred == center_id]
            density[center_id.item()] = item.mean() / np.log(len(item) + 10)

    dmax = density.max()
    for center_id in pred.unique():
        if (pred == center_id).nonzero().shape[0] <= 1:
            density[center_id.item()] = dmax
            
    density = density.cpu().numpy()

    density = density.clip(np.percentile(density, 10), np.percentile(density, 90)) #clamp extreme values for stability
    density = args.temperature * density / density.mean()  #scale the mean to temperature 
    density = torch.from_numpy(density).cuda()

    # centroids = torch.Tensor(centroids).cuda()
    centroids = F.normalize(centroids, p=2, dim=1)    
    results['centroids'].append(centroids)
    results['density'].append(density)
    results['im2cluster'].append(pred)    

    return results, km




@logger.catch
def train(projection_head, model, train_loader, test_loader, unlabelled_train_loader, pcl_loader, args):
    optimizer = SGD(list(projection_head.parameters()) + list(model.parameters()), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3,)

    sup_con_crit = SupConLoss()
    criterion1 = nn.CrossEntropyLoss() 
    best_test_acc_lab, best_test_acc_ubl, best_test_acc_all = 0, 0, 0

    epoch = 0
    if args.debug:
        args.logger.debug('Sanity check: before training starts')
        with torch.no_grad():
            args.logger.info('Testing on disjoint test set...')
            all_acc_test, old_acc_test, new_acc_test = test_kmeans(model, test_loader, pcl_loader,
                                                                   epoch=epoch, save_name='Test ACC',
                                                                   args=args)
    cluster_result = None
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        train_acc_record = AverageMeter()
        proto_acc_record = AverageMeter()
        if epoch >= args.warmup_epochs and args.enable_pcl and epoch % args.pcl_update_interval == 0:
            # compute prototype for each class  
            features, label, all_uq_idxs = extract_features(model, pcl_loader, args=args)

            features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
            # features = features.numpy()
            labeled_length = len(pcl_loader.dataset.labelled_dataset)
            
            l_feat = features[:labeled_length].detach().clone().cuda()
            u_feat = features[labeled_length:].detach().clone().cuda()
            # cat_feat = torch.cat((l_feat, u_feat), dim=0)

            l_targets = label[:labeled_length].cuda()

            if not isinstance(args.num_cluster, list):
                args.num_cluster = [args.num_cluster] # 100

            # placeholder for clustering result
            cluster_result = {'im2cluster':[],'centroids':[],'density':[]}
            for idx, _ in enumerate(args.num_cluster):
                # cluster_result_, cluster_km = run_kmeans(u_feat, l_feat, l_targets, args, index=idx)
                cluster_result_, cluster_km = split_and_merge_op(u_feat, l_feat, l_targets, args)
                # check label assign from km
                cluster_result['im2cluster'].extend(cluster_result_['im2cluster'])
                cluster_result['centroids'].extend(cluster_result_['centroids'])
                cluster_result['density'].extend(cluster_result_['density'])


        projection_head.train()
        model.train()
        for batch_idx, batch in enumerate(train_loader):

            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.to(device), mask_lab.to(device).bool()
            images = torch.cat(images, dim=0).to(device)

            # Extract features with base model
            features = model(images)
            global_feats = features
            global_feats = torch.nn.functional.normalize(global_feats, dim=-1)

            features = projection_head(features)

            features = torch.nn.functional.normalize(features, dim=-1)

            # Choose which instances to run the contrastive loss on
            if args.contrast_unlabel_only:
                # Contrastive loss only on unlabelled instances
                f1, f2 = [f[~mask_lab] for f in features.chunk(2)]
                con_feats = torch.cat([f1, f2], dim=0)
            else:
                # Contrastive loss for all examples
                con_feats = features

            contrastive_logits, contrastive_labels = info_nce_logits(features=con_feats, args=args)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            pstr, loss = '', 0

            f1, f2 = [f[mask_lab] for f in features.chunk(2)]
            sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            sup_con_labels = class_labels[mask_lab]

            sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)
            
            pstr += f'con_loss: {contrastive_loss.item():.4f} '
            pstr += f'supcon_loss: {sup_con_loss.item():.4f} '

            loss += (1 - args.sup_con_weight) * contrastive_loss + args.sup_con_weight * sup_con_loss
            
            f1, f2 = global_feats.chunk(2)
            # --------------------------------------
            if cluster_result is not None and epoch >= args.warmup_epochs and args.enable_pcl:  
                q, k = f1, f2

                if not isinstance(all_uq_idxs, list):
                    all_uq_idxs = all_uq_idxs.numpy().tolist()
                old_uq_idxs = uq_idxs.clone()
                uq_idxs = uq_idxs.numpy().tolist()
                uq_idxs = torch.from_numpy(np.array([all_uq_idxs.index(item) for item in uq_idxs])).long().cuda()
                
                if args.supcon_pcl:
                    loss_proto = proto_for_supcon_logits(q, k, cluster_result, uq_idxs, all_uq_idxs, sup_con_crit,)
                else:
                    # label assign on this will biased towards labeled classes
                    # NOTE: use pcl only unlabelled
                    proto_labels, proto_logits, proto_logits_k = prototypical_logits(q, k, cluster_result, old_uq_idxs, all_uq_idxs, use_all_proto=args.use_all_proto)

                    loss_proto = 0

                    for idx, (proto_out, proto_out_k, proto_target) in enumerate(zip(proto_logits, proto_logits_k, proto_labels)):
                        if idx == 0:
                            if proto_out.shape[0] == 0:
                                accp = 0
                            else:
                                
                                accp = accuracy(proto_out, proto_target)[0]
                            proto_acc_record.update(accp.item(), q.size(0))

                        if proto_out.shape[0] == 0:
                            loss_proto += 0.
                        else:
                            loss_proto += 0.5 * criterion1(proto_out, proto_target) + 0.5 * criterion1(proto_out_k, proto_target)
                            
                loss_proto = loss_proto / len(args.num_cluster) 

                loss += loss_proto 
                
                pstr += f'pcl_loss: {loss_proto.item():.4f} '

            # Train acc
            _, pred = contrastive_logits.max(1)
            acc = (pred == contrastive_labels).float().mean().item()
            train_acc_record.update(acc, pred.size(0))
            loss_record.update(loss.item(), class_labels.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % args.print_freq == 0 or (batch_idx + 1) % len(train_loader) == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

            if cluster_result is not None and epoch >= args.warmup_epochs and args.enable_pcl:  
                if args.momentum_proto:
                    with torch.no_grad():
                        cls_prototypes = cluster_result['centroids'][0]
                        for feat, lbl in zip(q, proto_labels[0]):
                            cls_prototypes[lbl] = args.momentum_proto * cls_prototypes[lbl] + (1 - args.momentum_proto) * feat
                        cls_prototypes = F.normalize(cls_prototypes, p=2, dim=1)
                        cluster_result['centroids'][0] = cls_prototypes

        # Step schedule
        exp_lr_scheduler.step()
        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} | Seen Class Acc: {:.4f} Proto Acc: {:.4f} '.format(epoch, loss_record.avg,
                                                                                  train_acc_record.avg, proto_acc_record.avg))

        with torch.no_grad():
            args.logger.info('Testing on disjoint test set...')
            if args.evaluate_with_proto and cluster_result is not None:
                all_acc_test, old_acc_test, new_acc_test = test_kmeans(model, test_loader, pcl_loader,
                                                                       epoch=epoch, save_name='Test ACC', cluster_res=cluster_result,
                                                                       args=args)
            else:
                all_acc_test, old_acc_test, new_acc_test = test_kmeans(model, test_loader, pcl_loader,
                                                                       epoch=epoch, save_name='Test ACC',
                                                                       args=args)

        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('Train Acc Labelled Data', train_acc_record.avg, epoch)
        args.writer.add_scalar('Proto ACC', proto_acc_record.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)

        args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))


        torch.save(model.state_dict(), args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))

        torch.save(projection_head.state_dict(), args.model_path[:-3] + '_proj_head.pt')
        args.logger.info("projection head saved to {}.".format(args.model_path[:-3] + '_proj_head.pt'))

        if new_acc_test > best_test_acc_ubl:
            args.logger.info(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')

            torch.save(model.state_dict(), args.model_path[:-3] + f'_best.pt')
            args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))
            torch.save(projection_head.state_dict(), args.model_path[:-3] + f'_proj_head_best.pt')
            args.logger.info("projection head saved to {}.".format(args.model_path[:-3] + f'_proj_head_best.pt'))
            # NOTE: save the prototypes for classification
            torch.save(cluster_result, args.model_path[:-3] + f'_prototype.pth')
            args.logger.info("prototypes saved to {}.".format(args.model_path[:-3] + f'_prototype.pth'))

            best_test_acc_lab, best_test_acc_ubl, best_test_acc_all = old_acc_test, new_acc_test, all_acc_test

        args.logger.info(f'Exp Name: {args.exp_name}')
        args.logger.info(f'Best metrics: All: {best_test_acc_all:.4f} Old: {best_test_acc_lab:.4f} New: {best_test_acc_ubl:.4f}')


def test_kmeans(model, test_loader, train_loader, epoch, save_name, args, cluster_res=None, ):
    model.eval()
    all_feats, targets, mask, proto_pred = [], [], np.array([]), []

    args.logger.info('Collating features from training set...')
    # First extract all features
    for batch_idx, (images, label, _, _) in enumerate(tqdm(train_loader)):
        images = images.cuda()
        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        if cluster_res is not None:
            preds = torch.mm(feats, cluster_res['centroids'][0].t())
            proto_pred.append(preds.cpu().numpy())

        all_feats.append(feats.cpu().numpy())
        targets.append(label.cpu().numpy()) # shouldn't be reading the labels for the test set
        # mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
        #                                  else False for x in label]))


    all_feats = np.concatenate(all_feats)
    targets = np.concatenate(targets)
    labeled_length = len(train_loader.dataset.labelled_dataset)
    
    l_feats = all_feats[:labeled_length]#.detach().clone().cuda()
    u_feats = all_feats[labeled_length:]#.detach().clone().cuda()
    l_targets = targets[:labeled_length]#.cuda()
    
    
    # __import__("ipdb").set_trace()

    args.logger.info('Collating features from testing set...')
    test_feats, test_labels, test_proto_pred = [], [], []
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda()
        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        if cluster_res is not None:
            preds = torch.mm(feats, cluster_res['centroids'][0].t())
            test_proto_pred.append(preds.cpu().numpy())

        test_feats.append(feats.cpu().numpy())
        test_labels.append(label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in args.train_classes
                                         else False for x in label]))

    test_feats = np.concatenate(test_feats)
    test_labels = np.concatenate(test_labels)
    

    if cluster_res is not None:
        preds = np.concatenate(test_proto_pred, )
        preds = np.argmax(preds, axis=1)
        
        s_score = silhouette_score(test_feats, preds, metric='cosine')
        
        args.logger.info(f'Silhouette score: {s_score:.6f}')
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=test_labels, y_pred=preds, mask=mask,
                                                        T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                        writer=args.writer, args=args)
        args.logger.info(f'{save_name} proto eval: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}')
        if args.disable_kmeans_eval:
            # NOTE: this cause old performance to degrade
            return all_acc, old_acc, new_acc

    if args.use_pca_testing:
        args.logger.info('Scores are evaluated using PCA')
        pca = PCA(n_components=128, whiten=True, random_state=1)
        pca.fit(all_feats)
        all_feats = pca.transform(all_feats)
        test_feats = pca.transform(test_feats)

    args.logger.info('Fitting K-Means...')
    num_clusters = len(set(args.train_classes) | set(args.unlabeled_classes))
    if args.use_faiss:
        raise NotImplementedError
        centroids, preds = kmeans_faiss(all_feats, k=num_clusters, verbose=True)
    elif args.use_sskmeans:
        
        # kmeans = K_Means(k=num_clusters, pairwise_batch_size=1024, tolerance=1e-4, use_gpu=True)
        kmeans = K_Means(k=num_clusters, pairwise_batch_size=1024, tolerance=1e-4)
        kmeans.fit_mix(u_feats, l_feats, l_targets)
        preds = kmeans.predict(test_feats)
        preds = preds.numpy()
    else:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_feats)
        preds = kmeans.predict(test_feats)
    args.logger.info('Done!')

    s_score = silhouette_score(test_feats, preds, metric='cosine')
    args.logger.info(f'Silhouette score: {s_score:.6f}')

    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=test_labels, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer, args=args)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2', 'ucd'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--model_arch', type=str, default='vit_base', help='which model to use')
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=False)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_best_thresh', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_con_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)
    
    parser.add_argument('--use_faiss', default=False, type=str2bool, help='use faiss for faster kmeans')
    parser.add_argument('--use_sskmeans', default=False, type=str2bool, help='use sskmeans')
    parser.add_argument('--use_mae', default=False, type=str2bool, help='use mae for pretrain')
    parser.add_argument('--use_ucd', default=False, type=str2bool, help='use ucd for everything')

    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    

    parser.add_argument('--pi_prior', default='uniform', type=str, help='')
    parser.add_argument('--prior_dir_counts', default=0.1, type=float)
    parser.add_argument('--prior_mu_0', default='data_mean', type=str)
    parser.add_argument('--prior_sigma_choice', default='isotropic', type=str)
    parser.add_argument('--prior_sigma_scale', default=.005, type=float)
    parser.add_argument('--prior_kappa', default=0.0001, type=float)
    parser.add_argument('--prior_nu', default=769, type=int)
    
    parser.add_argument('--enable_pcl', default=False, type=str2bool, )
    parser.add_argument('--proto_consistency', default=False, type=str2bool)
    parser.add_argument('--pcl_update_interval', default=1, type=int)
    parser.add_argument('--use_all_proto', default=True, type=str2bool)
    parser.add_argument('--pcl_only_unlabel', default=True, type=str2bool)
    parser.add_argument('--supcon_pcl', default=False, type=str2bool)
    parser.add_argument('--pcl_weight', default=1.0, type=float)
    
    parser.add_argument('--enable_proto_pair', default=False, type=str2bool)
    parser.add_argument('--pair_proto_num_multiplier', default=3, type=int)
    
    parser.add_argument('--topk', default=10, type=int)

    parser.add_argument('--evaluate_with_proto', default=False, type=str2bool)
    parser.add_argument('--disable_kmeans_eval', default=False, type=str2bool)
    
    parser.add_argument('--use_pca_testing', default=False, type=str2bool)
    
    # NOTE: strong and weak aug
    parser.add_argument('--use_strong_aug', default=False, type=str2bool)
    
    parser.add_argument('--momentum_proto', default=False, type=str2bool)
    parser.add_argument('--momentum_proto_weight', default=0.99, type=float)
    
    parser.add_argument('--me_max', default=False, type=str2bool)

    parser.add_argument('--exp_name', default=None, type=str)
    
    parser.add_argument('--plabel_correct', default=False, type=str2bool)
    parser.add_argument('--plabel_conf_thr', default=0.8, type=float)
    parser.add_argument('--plabel_metric_k_num', default=10, type=int)
    
    parser.add_argument('--debug', action='store_true',)

    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)
    print(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    if args.dataset_name in customize_datasets: 
        args.num_labeled_classes = len(args.train_classes)
        args.num_unlabeled_classes = len(set(args.unlabeled_classes) - set(args.train_classes))

    init_experiment(args, runner_name=['metric_learn_gcd'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')

    if args.base_model == 'vit_dino':
        args.interpolation = 3
        args.crop_pct = 0.875
        pretrain_path = dino_pretrain_path
        if args.use_mae:
            pretrain_path = mae_inat_pretrain_path

        model = vits.__dict__[args.model_arch]()

        state_dict = torch.load(pretrain_path, map_location='cpu')
        if args.use_mae:
            state_dict = state_dict['model']
        msg = model.load_state_dict(state_dict, strict=False)
        args.logger.info(msg)

        if args.warmup_model_dir is not None:
            args.logger.info(f'Loading weights from {args.warmup_model_dir}')
            model.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

        model.to(device)

        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3
        args.mlp_out_dim = 128

        for m in model.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True

    elif args.base_model == 'cnn':
        args.interpolation = 3
        args.crop_pct = 0.875
        args.image_size = 128 
        args.feat_dim = 256
        args.num_mlp_layers = 1
        args.mlp_out_dim = 128

        model = wrn.__dict__[args.model_arch]()
        model.to(device)
    else:
        raise NotImplementedError
    
    args.logger.info('model build')

    if not args.use_strong_aug:
        train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
        train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    else:
        train_transform, test_transform, strong_transform = get_transform(args.transform, image_size=args.image_size, args=args)
        train_transform = StrongWeakView(strong_transform, train_transform)

    # if args.enable_pcl:
    if args.use_strong_aug:
        pcl_transform, _, _ = get_transform('weak', image_size=args.image_size, args=args)
    else:
        pcl_transform, _ = get_transform('weak', image_size=args.image_size, args=args)

    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)
    
    # if args.enable_pcl:
    pcl_dataset, _, _, _ = get_datasets(args.dataset_name, pcl_transform, test_transform, args)

    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=args.batch_size, shuffle=False)
    
    # this will be used for evaluation
    pcl_loader = DataLoader(pcl_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,)

    if args.base_model == 'cnn':
        projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                                   out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers, hidden_dim=256)
    else:
        projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                                   out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)

    projection_head.to(device)

    train(projection_head, model, train_loader, test_loader_labelled, test_loader_unlabelled, pcl_loader, args)





