# The CP2_MoCo model is built upon moco v2 code base:
# https://github.com/facebookresearch/moco
# Copyright (c) Facebook, Inc. and its affilates. All Rights Reserved
import torch
import torch.nn as nn
from mmseg.models import build_segmentor

class CP2_MOCO(nn.Module):
    def __init__(self, cfg, dim=128, K=65536, m=0.999, T=0.2):
        super(CP2_MOCO, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = build_segmentor(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        self.encoder_k = build_segmentor(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        
        if ptr + batch_size > self.K:
            self.queue[:, ptr:self.K] = keys[0:self.K - ptr].T
            self.queue[:, 0:ptr + batch_size - self.K] = keys[self.K - ptr:batch_size].T
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, mask_q, mask_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        current_bs = im_q.size(0)

        mask_q = mask_q.reshape(current_bs, -1)
        mask_k = mask_k.reshape(current_bs, -1)

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxCx14x14
        q = q.reshape(q.shape[0], q.shape[1], -1)    # queries: NxCx196
        q_dense = nn.functional.normalize(q, dim=1)

        q_pos = nn.functional.normalize(torch.einsum('ncx,nx->nc', [q_dense, mask_q]), dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.encoder_k(im_k)  # keys: NxC
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

            k = k.reshape(k.shape[0], k.shape[1], -1)    # keys: NxCx196
            k_dense = nn.functional.normalize(k, dim=1)     # NxCx120
            k_pos = nn.functional.normalize(torch.einsum('ncx,nx->nc', [k_dense, mask_k]), dim=1)

        # dense logits
        logits_dense = torch.einsum('ncx,ncy->nxy', [q_dense, k_dense])     #Nx196x196
        labels_dense = torch.einsum('nx,ny->nxy', [mask_q, mask_k])
        labels_dense = labels_dense.reshape(labels_dense.shape[0], -1)
        mask_dense = torch.einsum('x,ny->nxy', [torch.ones(196).cuda(), mask_k])
        mask_dense = mask_dense.reshape(mask_dense.shape[0], -1)

        # moco logits
        l_pos = torch.einsum('nc,nc->n', [q_pos, k_pos]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q_pos, self.queue.clone().detach()])
        logits_moco = torch.cat([l_pos, l_neg], dim=1)
        labels_moco = torch.zeros(logits_moco.shape[0], dtype=torch.long).cuda()

        # apply temperature
        logits_moco /= self.T

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_pos)

        return logits_moco, logits_dense, labels_moco, labels_dense, mask_dense


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
