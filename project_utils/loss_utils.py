import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from tqdm.auto import tqdm



class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

class StrongWeakView(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, strong_transform, weak_transform):
        self.strong_transform = strong_transform
        self.weak_transform = weak_transform

    def __call__(self, x):
        return [self.weak_transform(x), self.strong_transform(x)]

pdist = nn.PairwiseDistance(2)

def pairwise_NNs_inner(x):
    """
    Pairwise nearest neighbors for L2-normalized vectors.
    Uses Torch rather than Faiss to remain on GPU.
    """
    # parwise dot products (= inverse distance)
    dots = torch.mm(x, x.t())
    n = x.shape[0]
    dots.view(-1)[::(n+1)].fill_(-1)  # Trick to fill diagonal with -1
    _, I = torch.max(dots, 1)  # max inner prod -> min distance
    return I

# entropy loss
# I = pairwise_NNs_inner(ins.data)
# distances = pdist(ins, ins[I])
# loss_uniform = - torch.log(n * distances).mean()


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



def info_nce_logits(features, args, device='cuda'):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature
    return logits, labels


def proto_for_supcon_logits(q, k, cluster_result, uq_idxs, all_uq_idxs, sup_con):
    feats = torch.cat([q.unsqueeze(1), k.unsqueeze(1)], dim=1)
    
    supcon_loss = 0
    for idx, (im2cluster,prototypes,density) in enumerate(zip(cluster_result['im2cluster'],
                                                              cluster_result['centroids'], 
                                                              cluster_result['density'])):
        supcon_loss = supcon_loss + sup_con(feats, labels=im2cluster[uq_idxs])

    return supcon_loss
    



def prototypical_logits(q, k, cluster_result, uq_idxs, all_uq_idxs, use_all_proto=True):
    # uq_idxs is used to do label assign
    proto_labels, proto_logits, proto_logits_k = [], [], []
    
    if not isinstance(all_uq_idxs, list):
        all_uq_idxs = all_uq_idxs.numpy().tolist()
    uq_idxs = uq_idxs.numpy().tolist()
    uq_idxs = torch.from_numpy(np.array([all_uq_idxs.index(item) for item in uq_idxs])).long().cuda()

    for idx, (im2cluster,prototypes,density) in enumerate(zip(cluster_result['im2cluster'],
                                                            cluster_result['centroids'],
                                                            cluster_result['density'])):
        if idx == 0 and use_all_proto:
            # NOTE: we don't do sample for negative for the cluster used for classify
            # this may cause old class performance to degrade.
            logits_proto = torch.mm(q, prototypes.t())
            logits_proto_k = torch.mm(k, prototypes.t())
            pos_proto_id = im2cluster[uq_idxs]
            labels_proto = pos_proto_id
            temp_proto = density[torch.arange(prototypes.shape[0]).long().cuda()]
            logits_proto /= temp_proto
            logits_proto_k /= temp_proto
            
        else:
            # get positive prototypes
            pos_proto_id = im2cluster[uq_idxs]
            pos_prototypes = prototypes[pos_proto_id]    
            
            # sample negative prototypes
            all_proto_id = [i for i in range(im2cluster.max())]       
            neg_proto_id = set(all_proto_id) - set(pos_proto_id.tolist())
            # neg_proto_id = sample(neg_proto_id, args.r) #sample r negative prototypes default: 100
            neg_proto_id = list(neg_proto_id)
            neg_prototypes = prototypes[neg_proto_id]    

            proto_selected = torch.cat([pos_prototypes, neg_prototypes],dim=0)
            
            # compute prototypical logits
            logits_proto = torch.mm(q, proto_selected.t())
            logits_proto_k = torch.mm(k, proto_selected.t())
            
            # targets for prototype assignment
            labels_proto = torch.linspace(0, q.size(0)-1, steps=q.size(0)).long().cuda()
            
            # scaling temperatures for the selected prototypes
            temp_proto = density[torch.cat([pos_proto_id, torch.LongTensor(neg_proto_id).cuda()],dim=0)]  
            logits_proto /= temp_proto
            logits_proto_k /= temp_proto
            
        proto_labels.append(labels_proto)
        proto_logits.append(logits_proto)
        proto_logits_k.append(logits_proto_k)
    return proto_labels, proto_logits, proto_logits_k 


@torch.no_grad()
def extract_metric(net, p_label, evalloader, n_num, num_classes, all_uq_idxs):
    # NOTE this thing can work and improve the pseudo labels
    # TODO: refactor to do only one forward pass
    if not isinstance(all_uq_idxs, list):
        all_uq_idxs = all_uq_idxs.numpy().tolist()
    net.eval()
    feature_bank = []
    for batch_idx, (inputs1, _, indexes, _) in enumerate(tqdm(evalloader)):
        out = net(inputs1.cuda())
        feature_bank.append(out)
    feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
    sim_indices_list = []
    for batch_idx, (inputs1, _, indexes, _) in enumerate(tqdm(evalloader)):
        out = net(inputs1.cuda(non_blocking=True))
        sim_matrix = torch.mm(out, feature_bank)
        _, sim_indices = sim_matrix.topk(k=n_num, dim=-1)
        sim_indices_list.append(sim_indices)
    feature_labels = p_label.cuda()
    first = True
    count = 0
    for batch_idx, (inputs1, _, indexes, _) in enumerate(tqdm(evalloader)):
        indexes = indexes.numpy().tolist()
        indexes = torch.from_numpy(np.array([all_uq_idxs.index(item) for item in indexes])).long().cuda()

        labels = p_label[indexes].cuda().long()
        sim_indices = sim_indices_list[count]
        sim_labels = torch.gather(feature_labels.expand(inputs1.size(0), -1), dim=-1, index=sim_indices)
        # counts for each class
        one_hot_label = torch.zeros(inputs1.size(0) * sim_indices.size(1), num_classes).cuda()
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
        pred_scores = torch.sum(one_hot_label.view(inputs1.size(0), -1, num_classes), dim=1)
        count += 1
        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        prob, _ = torch.max(F.softmax(pred_scores, dim=-1), 1)   
        # Check whether prediction and current label are same
        s_idx1 = (pred_labels[:, :1].float() == labels.unsqueeze(dim=-1).float()).any(dim=-1).float()
        s_idx = (s_idx1 == 1.0)

        if first:
            prob_set = prob
            pred_same_label_set = s_idx
            first = False
        else:
            prob_set = torch.cat((prob_set, prob), dim = 0)
            pred_same_label_set = torch.cat((pred_same_label_set, s_idx), dim = 0)

    return pred_same_label_set
            
@torch.no_grad()
def extract_confidence(net, prototypes, p_label, evalloader, threshold, all_uq_idxs):
    net.eval()
    if not isinstance(all_uq_idxs, list):
        all_uq_idxs = all_uq_idxs.numpy().tolist()
    devide = torch.tensor([]).cuda()
    for batch_idx, (inputs1, _, indexes, _) in enumerate(tqdm(evalloader)):
        inputs1 = inputs1.cuda()
        indexes = indexes.numpy().tolist()
        indexes = torch.from_numpy(np.array([all_uq_idxs.index(item) for item in indexes])).long().cuda()
        logits = net(inputs1)
        logits = logits @ prototypes.t()
        prob = torch.softmax(logits, dim=-1)
        max_probs, _ = torch.max(prob, dim=-1)
        mask = max_probs.ge(threshold).float()
        devide = torch.cat([devide, mask])
    
    return devide


@torch.no_grad()
def extract_feats_for_check(net, prototypes, evalloader):
    # forward to the model
    # get features, batch of features, batch of sims, batch of indexs
    net.eval()
    feats, sims, idxes, logit = [], [], [], []
    for batch_idx, (inputs1, _, indexes, _) in enumerate(tqdm(evalloader)):
        inputs1 = inputs1.cuda()
        feat = net(inputs1)
        logits = feat @ prototypes.t()
        feats.append(feat.cpu())
        logit.append(logits.cpu())
        idxes.append(indexes)
    features = torch.cat(feats, dim=0)
    for feat in feats:
        sim = torch.mm(feat, features.t())
        sims.append(sim)

    return feats, sims, idxes, logit

@torch.no_grad()
def extract_items_for_check_with_feats(feats, idxes, prototypes, ):
    sims, logit = [], []
    for feat, indexes in zip(feats, idxes):
        logits = feat.to(prototypes.device) @ prototypes.t()
        logit.append(logits.cpu())
    features = torch.cat(feats, dim=0)
    for feat in feats:
        sim = torch.mm(feat, features.t())
        sims.append(sim)
    return feats, sims, idxes, logit

def conf_check(logits, threshold):
    divide = torch.tensor([])
    for logit in logits:
        prob = torch.softmax(logit, dim=-1)
        max_probs, _ = torch.max(prob, dim=-1)
        mask = max_probs.ge(threshold).float()
        divide = torch.cat([divide, mask])
    return divide

def metric_check(p_label, sims, idxs, num_classes, n_num, all_uq_idxs,):
    count = 0
    if not isinstance(all_uq_idxs, list):
        all_uq_idxs = all_uq_idxs.numpy().tolist()

    sim_indices_list = []
    for sim_matrix in sims:
        _, sim_indices = sim_matrix.topk(k=n_num, dim=-1)
        sim_indices_list.append(sim_indices)

    feature_labels = p_label
    first = True
    count = 0
    for indexes in idxs:
        indexes = indexes.numpy().tolist()
        indexes = torch.from_numpy(np.array([all_uq_idxs.index(item) for item in indexes])).long()

        labels = p_label[indexes].long()
        sim_indices = sim_indices_list[count]
        sim_labels = torch.gather(feature_labels.expand(indexes.size(0), -1), dim=-1, index=sim_indices)
        # counts for each class
        one_hot_label = torch.zeros(indexes.size(0) * sim_indices.size(1), num_classes)
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
        pred_scores = torch.sum(one_hot_label.view(indexes.size(0), -1, num_classes), dim=1)
        count += 1
        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        prob, _ = torch.max(F.softmax(pred_scores, dim=-1), 1)   
        # Check whether prediction and current label are same
        s_idx1 = (pred_labels[:, :1].float() == labels.unsqueeze(dim=-1).float()).any(dim=-1).float()
        s_idx = (s_idx1 == 1.0)

        if first:
            prob_set = prob
            pred_same_label_set = s_idx
            first = False
        else:
            prob_set = torch.cat((prob_set, prob), dim = 0)
            pred_same_label_set = torch.cat((pred_same_label_set, s_idx), dim = 0)

    return pred_same_label_set
            

