import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

eps = 1e-7
import torch.distributed as dist

import wandb
def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


def torch_qr(a, mode='complete', out=None, gram='classical'):
    """
    Due to a bug in MAGMA, qr on cuda is super slow for small matrices. 
    Therefore, this step must be performed on the cpu.
    
    This function aims to provide a temporary relief for using 
    `torch.linalg.qr` on GPU by implementing a Gram-Schmidt process. 
    
    Note: This implementation does not support backward propagation, and 
          only supports the 'complete' mode.
    
    See the following regarding this Bug:
        https://github.com/pytorch/pytorch/issues/22573
        https://github.com/cornellius-gp/gpytorch/pull/1224
        
    The input arguments, other than 'gram', follow the PyTorch standard. 
    See the following for their definition:
        https://pytorch.org/docs/stable/generated/torch.linalg.qr.html
        
    Parameters
    ----------
    a: (torch.tensor) the input tensor. Must have a shape of 
        `(*mb_dims, dim, dim)`, where `mb_dims` shows the batch 
        dimensions.

    mode: (str) Either `'complete'` or `'reduced'`. This current 
        implementation only supports the former.
        
    out: (None or torch.tensor) The output tensor for the `Q` matrix. 
        If provided, must have the same shape as `a`.
        
    gram: (str) The Gram-Schmidt process variant. 
    
        * The `classical` variant makes `O(dim)` calls to CUDA 
          and can be more efficient. 
          
        * The `modified` variant can be slightly more accurate, 
          but makes CUDA `O(dim^2)` calls and thus is less efficient.
          
          See Section 14.2 of "Numerical Linear Algebra with Applications" 
          by William Ford on the numerical stability of Gram-Schmidt and 
          its modified variant:
          
          https://www.sciencedirect.com/science/article/abs/pii/B9780123944351000144
          
        * The `cpu` variant uses Pytorch's routine on CPU.
          
        This has to be one of `('classical', 'modified', 'cpu')`.
        
    Output
    ------
    q: (torch.tensor) The output orthonormal matrix. 
        This should have a shape of `(*mb_dims, dim, dim)`.
    
    r: (torch.tensor) The output upper triangle matrix. 
        This should have a shape of `(*mb_dims, dim, dim)`.
    """

    assert not a.requires_grad

    # First Solution: Performing the QR decomposition on CPU
    # Issues: 
    #    1. Pytorch may still only utilize one thread 
    #       practically even though `torch.get_num_threads()` 
    #       may be large.
    #    2. Reliance on CPU resources.
    if gram == 'cpu':
        q, r = torch.linalg.qr(a.detach().cpu(), mode=mode, out=out)
        return q.to(device=a.device), r.to(device=a.device)
    
    ###############################################################
    ################## Initializing & Identifying #################
    ###############################################################
    assert mode == 'complete', 'reduced is not implemented yet'
    # The bactch dimensions
    mb_dims = a.shape[:-2]
    # The input device
    tch_device = a.device
    
    # The Data Type for performing the mathematical caculations
    # Note: Gram-schmidt is numerically unstable. For this reason, even 
    # when the input may be float32, we will do everything in float64.
    tch_dtype = torch.float64
    
    # The QR process dimension
    dim = a.shape[-1]
    assert a.shape == (*mb_dims, dim, dim)

    if out is None:
        q = torch.empty(*mb_dims, dim, dim, device=tch_device, dtype=tch_dtype)
    else:
        q = out
    assert q.shape == (*mb_dims, dim, dim)
    
    # Casting the `a` input to `tch_dtype` and using it from now on
    a_f64 = a.to(dtype=tch_dtype)
    
    ###############################################################
    ################### Performing Gram-Schmidt ###################
    ###############################################################
    if gram == 'classical':
        # Performing the classical Gram-Schmidt Process.
        
        # Creating a copy of `a` to avoid messing up the original input
        acp = a_f64.detach().clone()
        assert acp.shape == (*mb_dims, dim, dim)
        
        for k in range(dim):
            qk_unnorm = acp[..., :, k:k+1]
            assert qk_unnorm.shape == (*mb_dims, dim, 1)

            qk = qk_unnorm / qk_unnorm.norm(dim=-2, keepdim=True)
            assert qk.shape == (*mb_dims, dim, 1)

            a_qkcomps = qk.reshape(*mb_dims, 1, dim).matmul(acp)
            assert a_qkcomps.shape == (*mb_dims, 1, dim)

            # Removing the `qk` components from `a`
            acp -= qk.matmul(a_qkcomps)
            assert acp.shape == (*mb_dims, dim, dim)

            q[..., :, k] = qk.reshape(*mb_dims, dim)
    elif gram == 'modified':
        # Performing the modified Gram-Schmidt Process.
        for i in range(dim):
            q[..., i] = a_f64[..., i]
            for j in range(i):
                err_ij = torch.einsum('...i,...i->...', q[..., j], q[..., i])
                assert err_ij.shape == (*mb_dims,)
                q[..., i] -=  err_ij.reshape(*mb_dims, 1) * q[..., j]
            q[..., i] /= q[..., i].norm(dim=-1, keepdim=True)
    else:
        raise ValueError(f'Unknown gram={gram}')

    r = q.transpose(-1, -2).matmul(a_f64)
    assert r.shape == (*mb_dims, dim, dim)

    ###############################################################
    ######################## Final Cleanup ########################
    ###############################################################
    # Making sure the lower triangle of `r` is absolutely zero!
    col = torch.arange(dim, device=tch_device, dtype=tch_dtype).reshape(1, dim)
    assert col.shape == (1, dim)

    row = col.reshape(dim, 1)
    assert row.shape == (dim, 1)
    
    mb_ones = [1] * len(mb_dims)
    r *= (row <= col).reshape(*mb_ones, dim, dim)
    
    # Casting the `q` and `r` outputs to the `a` input dtype for compatibility
    q_out, r_out = q.to(dtype=a.dtype), r.to(dtype=a.dtype)
    
    return q_out, r_out


def calculate_lambda_max_loss(x):   
    # (batch_positions, d, n)  
    import time
    
    if torch.isnan(x).any():
        raise ValueError(f"NaNs detected in clients_tensor before normalization.")

    x = F.normalize(x, p=2, dim=-1).to('cpu')


    if torch.isnan(x).any():
        raise ValueError(f"NaNs detected in clients_tensor after normalization.")

    A = x.permute(1, 2, 0).contiguous()   
    eps = 1e-6

    start_time = time.time()
    Q, R = torch.linalg.qr(A, mode="reduced")
    end_time = time.time()
        
    r_diag = R.abs().diagonal(dim1=-2, dim2=-1)           # (E, min(d,B))
    k      = (r_diag > eps).sum(dim=1)                    # (E,)
    cols   = torch.arange(Q.size(-1), device=Q.device)    # (d,)
    mask   = cols[None, None, :] < k[:, None, None]       # (E, 1, d)
    Qm     = Q * mask                                     
    projs  = Qm @ Qm.transpose(-2, -1) 
    avg_proj    = projs.mean(dim=0) 

    eigvals = torch.linalg.eigvalsh(avg_proj)
    lambda_max = eigvals[-1]
    return lambda_max.to('cuda')

class CosineDiversityLoss(nn.Module):
    """
    Penalize experts whose outputs are too similar:
    loss = mean_{i<j} [ cosine_similarity(logits_i, logits_j) ]
    """
    def __init__(self, weight: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, logits_list):
        # logits_list: list of (N, C) tensors from each expert
        M = len(logits_list)
        if M < 2 or self.weight == 0:
            return logits_list[0].new_tensor(0.)

        # 1) stack into (M, N, C)
        # x = torch.stack(logits_list, dim=0)

        # 2) normalize along the class-dimension
        x = F.normalize(logits_list, p=2, dim=2, eps=self.eps)  # still (M, N, C)

        # 3) compute pairwise cosine-sims per sample: (M, M, N)
        sims = torch.einsum('mnc,knc->mnk', x, x)

        # 4) average over the batch‐dimension N → (M, M)
        sims = sims.mean(dim=2)

        # 5) take only the upper‐triangle entries i<j
        idx_i, idx_j = torch.triu_indices(M, M, offset=1, device=sims.device)
        pair_sims = sims[idx_i, idx_j]  # shape = (M*(M-1)/2,)

        # 6) mean over all pairs
        loss = torch.abs(pair_sims.mean())

        wandb.log({"cosine_diversity_loss": loss.item()}, commit=False)

        return self.weight * loss



class FocalLoss(nn.Module):
    def __init__(self, cls_num_list=None, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def _hook_before_epoch(self, epoch):
        pass

    def forward(self, output_logits, target):
        return focal_loss(F.cross_entropy(output_logits, target, reduction='none', weight=self.weight), self.gamma)


class CrossEntropyLoss(nn.Module):
    def __init__(self, cls_num_list=None, reweight_CE=False):
        super().__init__()
        if reweight_CE:
            idx = 1  # condition could be put in order to set idx
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        else:
            self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.per_cls_weights is not None:
            self.per_cls_weights = self.per_cls_weights.to(device)

        return self

    def forward(self, output_logits, target):  # output is logits
        return F.cross_entropy(output_logits, target, weight=self.per_cls_weights)


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, reweight_epoch=-1):
        super().__init__()
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.
            self.m_list = None
        else:
            self.reweight_epoch = reweight_epoch
            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            assert s > 0
            self.s = s
            if reweight_epoch != -1:
                # CB loss
                idx = 1  # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)  # * class number
                # the effect of per_cls_weights / np.sum(per_cls_weights) can be described in the learning rate so the math formulation keeps the same.
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None
                self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)  # one-hot index

        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))

        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target):
        if self.m_list is None:
            return F.cross_entropy(output_logits, target)

        final_output = self.get_final_output(output_logits, target)
        return F.cross_entropy(final_output, target, weight=self.per_cls_weights)


class RIDELoss(nn.Module):
    def __init__(self, cls_num_list=None, base_diversity_temperature=1.0, max_m=0.5, s=30, reweight=True,
                 reweight_epoch=-1,
                 base_loss_factor=1.0, additional_diversity_factor=-0.2, reweight_factor=0.05):
        super().__init__()
        self.base_loss = F.cross_entropy
        self.base_loss_factor = base_loss_factor
        if not reweight:
            self.reweight_epoch = -1
        else:
            self.reweight_epoch = reweight_epoch

        # LDAM is a variant of cross entropy and we handle it with self.m_list.
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.

            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights_enabled_diversity = None
        else:
            # We will use LDAM loss if we provide cls_num_list.

            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            self.s = s
            assert s > 0

            if reweight_epoch != -1:
                idx = 1  # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float,
                                                            requires_grad=False)  # 这个是logits时算CE loss的weight
            else:
                self.per_cls_weights_enabled = None

            cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
            C = len(cls_num_list)  # class number
            per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor  # Eq.3

            # Experimental normalization: This is for easier hyperparam tuning, the effect can be described in the learning rate so the math formulation keeps the same.
            # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
            per_cls_weights = per_cls_weights / np.max(
                per_cls_weights)  # the effect can be described in the learning rate so the math formulation keeps the same.

            assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
            # save diversity per_cls_weights
            self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float,
                                                                  requires_grad=False).cuda()  # 这个是logits时算diversity loss的weight

        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))

        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)

        loss = 0

        # Adding RIDE Individual Loss for each expert
        for logits_item in extra_info['logits']:
            ride_loss_logits = output_logits if self.additional_diversity_factor == 0 else logits_item
            if self.m_list is None:
                loss += self.base_loss_factor * self.base_loss(ride_loss_logits, target)
            else:
                final_output = self.get_final_output(ride_loss_logits, target)
                loss += self.base_loss_factor * self.base_loss(final_output, target, weight=self.per_cls_weights_base)

            base_diversity_temperature = self.base_diversity_temperature

            if self.per_cls_weights_diversity is not None:
                diversity_temperature = base_diversity_temperature * self.per_cls_weights_diversity.view((1, -1))
                temperature_mean = diversity_temperature.mean().item()
            else:
                diversity_temperature = base_diversity_temperature
                temperature_mean = base_diversity_temperature

            output_dist = F.log_softmax(logits_item / diversity_temperature, dim=1)
            with torch.no_grad():
                # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
                mean_output_dist = F.softmax(output_logits / diversity_temperature, dim=1)

            loss += self.additional_diversity_factor * temperature_mean * temperature_mean * F.kl_div(output_dist,
                                                                                                      mean_output_dist,
                                                                                                      reduction='batchmean')

        return loss


def dkd_loss(logits_student, logits_teacher, target, alpha=1, beta=8, temperature=4):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    # pdb.set_trace()

    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')
            * (temperature ** 2)
        # / target.shape[0]
    )

    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean')
            * (temperature ** 2)
        # / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return t2



class MDCSLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, tau=2, use_cosine_loss=False, use_lambda_max=True):
        super().__init__()
        self.base_loss = F.cross_entropy
        self.cosine_loss = CosineDiversityLoss(weight=1.0)
        prior = np.array(cls_num_list) #/ np.sum(cls_num_list)

        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)  # class number
        self.s = s
        self.tau = 2

        self.use_cosine_loss = use_cosine_loss
        self.use_lambda_max = use_lambda_max

        self.additional_diversity_factor = -0.2
        out_dim = 100
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center1", torch.zeros(1, out_dim))
        self.center_momentum = 0.9
        self.warmup = 20  
        self.reweight_epoch = 200
        if self.reweight_epoch != -1:
            idx = 1  # condition could be put in order to set idx
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float,
                                                        requires_grad=False)  # 这个是logits时算CE loss的weight
        self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float,
                                                              requires_grad=False).cuda()  # 这个是logits时算diversity loss的weight



    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)  # output_logits indicates the final prediction

        loss = 0
        temperature_mean = 1
        temperature = 1  
        # Obtain logits from each expert
        epoch = extra_info['epoch']
        num = int(target.shape[0] / 2)

        expert1_logits = extra_info['logits'][0] + torch.log(torch.pow(self.prior, -0.5) + 1e-9)      #head

        expert2_logits = extra_info['logits'][1] + torch.log(torch.pow(self.prior, 1) + 1e-9)         #medium

        expert3_logits = extra_info['logits'][2] + torch.log(torch.pow(self.prior, 2.5) + 1e-9)       #few


        if self.use_cosine_loss:
            logits_list = extra_info['logits']
            loss += self.cosine_loss(logits_list)
        if self.use_lambda_max:
            logits_list = extra_info['logits']
            lambda_max = calculate_lambda_max_loss(logits_list)
            loss += lambda_max * 10.0

        teacher_expert1_logits = expert1_logits[:num, :]  # view1
        student_expert1_logits = expert1_logits[num:, :]  # view2

        teacher_expert2_logits = expert2_logits[:num, :]  # view1
        student_expert2_logits = expert2_logits[num:, :]  # view2

        teacher_expert3_logits = expert3_logits[:num, :]  # view1
        student_expert3_logits = expert3_logits[num:, :]  # view2




        teacher_expert1_softmax = F.softmax((teacher_expert1_logits) / temperature, dim=1).detach()
        student_expert1_softmax = F.log_softmax(student_expert1_logits / temperature, dim=1)

        teacher_expert2_softmax = F.softmax((teacher_expert2_logits) / temperature, dim=1).detach()
        student_expert2_softmax = F.log_softmax(student_expert2_logits / temperature, dim=1)

        teacher_expert3_softmax = F.softmax((teacher_expert3_logits) / temperature, dim=1).detach()
        student_expert3_softmax = F.log_softmax(student_expert3_logits / temperature, dim=1)


         

        teacher1_max, teacher1_index = torch.max(F.softmax((teacher_expert1_logits), dim=1).detach(), dim=1)
        student1_max, student1_index = torch.max(F.softmax((student_expert1_logits), dim=1).detach(), dim=1)

        teacher2_max, teacher2_index = torch.max(F.softmax((teacher_expert2_logits), dim=1).detach(), dim=1)
        student2_max, student2_index = torch.max(F.softmax((student_expert2_logits), dim=1).detach(), dim=1)

        teacher3_max, teacher3_index = torch.max(F.softmax((teacher_expert3_logits), dim=1).detach(), dim=1)
        student3_max, student3_index = torch.max(F.softmax((student_expert3_logits), dim=1).detach(), dim=1)


        # distillation
        partial_target = target[:num]
        kl_loss = 0
        if torch.sum((teacher1_index == partial_target)) > 0:
            kl_loss = kl_loss + F.kl_div(student_expert1_softmax[(teacher1_index == partial_target)],
                                         teacher_expert1_softmax[(teacher1_index == partial_target)],
                                         reduction='batchmean') * (temperature ** 2)

        if torch.sum((teacher2_index == partial_target)) > 0:
            kl_loss = kl_loss + F.kl_div(student_expert2_softmax[(teacher2_index == partial_target)],
                                         teacher_expert2_softmax[(teacher2_index == partial_target)],
                                         reduction='batchmean') * (temperature ** 2)

        if torch.sum((teacher3_index == partial_target)) > 0:
            kl_loss = kl_loss + F.kl_div(student_expert3_softmax[(teacher3_index == partial_target)],
                                         teacher_expert3_softmax[(teacher3_index == partial_target)],
                                         reduction='batchmean') * (temperature ** 2)

        # loss = loss + 0.6 * kl_loss * min(extra_info['epoch'] / self.warmup, 1.0)



        # expert 1
        loss += self.base_loss(expert1_logits, target)

        # expert 2
        loss += self.base_loss(expert2_logits, target)

        # expert 3
        loss += self.base_loss(expert3_logits, target)



        return loss

    @torch.no_grad()
    def update_center(self, center, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output))  # * dist.get_world_size())

        # ema update

        return center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINOLoss(nn.Module):
    def __init__(self, out_dim=65536, ncrops=4, warmup_teacher_temp=0.04, teacher_temp=0.04,
                 warmup_teacher_temp_epochs=10, nepochs=200, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
