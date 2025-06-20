# From https://github.com/kaidic/LDAM-DRW/blob/master/models/resnet_cifar.py
'''
Properly implemented ResNet for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
import pdb
import random

__all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.planes = planes
                self.in_planes = in_planes
                # self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (planes - in_planes) // 2, (planes - in_planes) // 2), "constant", 0))
                
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_s(nn.Module):

    def __init__(self, block, num_blocks, num_experts, num_classes=10, reduce_dimension=False, layer2_output_dim=None, layer3_output_dim=None, use_norm=False, returns_feat=True, use_experts=None, s=30, project=False, orthonormalise=True):
        super(ResNet_s, self).__init__()
        
        self.in_planes = 16
        self.num_experts = num_experts
        self.project=project
        self.orthonormalise = orthonormalise
        if self.project:
            print("Projecting to unique subspaces")
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.in_planes = self.next_in_planes

        if layer2_output_dim is None:
            if reduce_dimension:
                layer2_output_dim = 24
            else:
                layer2_output_dim = 32

        if layer3_output_dim is None:
            if reduce_dimension:
                layer3_output_dim = 48
            else:
                layer3_output_dim = 64

        self.layer2s = nn.ModuleList([self._make_layer(block, layer2_output_dim, num_blocks[1], stride=2) for _ in range(num_experts)])
        self.in_planes = self.next_in_planes
        self.layer3s = nn.ModuleList([self._make_layer(block, layer3_output_dim, num_blocks[2], stride=2) for _ in range(num_experts)])
        self.in_planes = self.next_in_planes
        self.projection_matrix = None

        if use_norm:
            self.linears = nn.ModuleList([NormedLinear(layer3_output_dim, num_classes) for _ in range(num_experts)])
            self.linear_head =  NormedLinear(layer3_output_dim, num_classes)  
            self.linear_few = NormedLinear(layer3_output_dim, num_classes)  
            self.mlp =  nn.Sequential(NormedLinear(16, 16), 
                                      nn.ReLU(),
                                      NormedLinear(16, 3))
        else:
            self.linears = nn.ModuleList([nn.Linear(layer3_output_dim, num_classes) for _ in range(num_experts)])
            self.linear_head =  nn.Linear(layer3_output_dim, num_classes)  
            self.linear_few = nn.Linear(layer3_output_dim, num_classes)  
            self.mlp =  nn.Sequential(nn.Linear(16, 16), 
                                      nn.ReLU(),
                                      nn.Linear(16, 3))
            s = 1

        if use_experts is None:
            self.use_experts = list(range(num_experts))
        elif use_experts == "rand":
            self.use_experts = None
        else:
            self.use_experts = [int(item) for item in use_experts.split(",")]

        self.s = s
        self.returns_feat = returns_feat
        self.apply(_weights_init)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        self.next_in_planes = self.in_planes
        for stride in strides:
            layers.append(block(self.next_in_planes, planes, stride))
            self.next_in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"
        count = 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.weight.requires_grad == False:
                    module.eval()
                    count += 1

        if count > 0:
            print("Warning: detected at least one frozen BN, set them to eval state. Count:", count)

    def _separate_part(self, x, ind):
        out = x
        out = (self.layer2s[ind])(out)
        out = (self.layer3s[ind])(out)
        self.feat_before_GAP.append(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        self.feat.append(out)
        out = (self.linears[ind])(out)
        out = out * self.s
        return out

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        
        outs = []
        self.feat = []
        self.logits = outs
        self.feat_before_GAP = []
        
        if self.use_experts is None:
            use_experts = random.sample(range(self.num_experts), self.num_experts - 1)
        else:
            use_experts = self.use_experts
        
        for ind in use_experts:
            outs.append(self._separate_part(out, ind))

        self.feat = torch.stack(self.feat, dim=1)
        self.feat_before_GAP = torch.stack(self.feat_before_GAP, dim=1)
        final_out = torch.stack(outs, dim=1)



        if self.project:
            # batch, K, dim = 256, 4, 100
            # U = torch.randn(batch, K, dim, device="cuda")
            # V = project_to_unique_subspaces(U, torch.randn(dim, dim, device="cuda"))
            # λ = calculate_lambda_max_loss(V)
            # print("λₘₐₓ on perfect orthogonal split:", λ.item())
            import math
            if self.projection_matrix is None:
                self.projection_matrix = torch.zeros((final_out.shape[-1], final_out.shape[-1]), device='cuda')
                torch.nn.init.kaiming_uniform_(self.projection_matrix, a=math.sqrt(5))
            final_out = project_to_unique_subspaces(
                final_out,
                self.projection_matrix
            )
        elif self.orthonormalise:
            projected_final_out = gram_schmidt_orthonormalise(final_out)
            projected_final_out[:,-1, :] = final_out[:,-1, :]
            final_out = projected_final_out
        mean_final_out = final_out.mean(dim=1)

        

        if self.returns_feat:
            return {
                "output": mean_final_out, 
                "feat": self.feat,
                "logits": final_out
            }
        else:
            return final_out

def resnet20():
    return ResNet_s(BasicBlock, [3, 3, 3])


def resnet32(num_classes=10, use_norm=False):
    return ResNet_s(BasicBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm)


def resnet44():
    return ResNet_s(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet_s(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet_s(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet_s(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


def batch_project_to_unique_subspaces(
    U: torch.Tensor,
    A: torch.Tensor
) -> torch.Tensor:
    batch, K, dim = U.shape
    # (batch, dim*dim)
    # 2) form a skew-symmetric S(x)
    S = A - A.transpose(-1,-2)             # (batch, dim, dim)
    I = torch.eye(dim, device=U.device).unsqueeze(0)  # (1,dim,dim)

    # 3) Cayley transform per-sample
    #    Q(x) = (I - S)^{-1}(I + S)
    Q = torch.linalg.solve(I - S, I + S)           # (batch, dim, dim)

    # 4) slice into K disjoint blocks and project each expert
    dsub = dim // U.shape[1]
    V = []
    for i in range(U.shape[1]):
        Bi = Q[:, :, i*dsub:(i+1)*dsub]            # (batch, dim, dsub)
        ui = U[:, i].unsqueeze(-1)                 # (batch, dim, 1)
        coords = Bi.transpose(-1,-2) @ ui          # (batch, dsub, 1)
        vi = Bi @ coords                           # (batch, dim, 1)
        V.append(vi.squeeze(-1))                   # (batch, dim)
    V = torch.stack(V, dim=1)                     # (batch, K, dim)
    return V




def project_to_unique_subspaces(
    U: torch.Tensor,
    A: torch.Tensor
) -> torch.Tensor:
    """
    Args:
      U: (batch, K, dim)                — MoE outputs
      A: (dim, dim)                     — unconstrained parameter
    Returns:
      V: (batch, K, dim)                — each expert in its own orthogonal subspace
    """
    batch, K, dim = U.shape
    base, rem = divmod(dim, K)      # e.g. for dim=100, K=6 → base=16, rem=4
    # first `rem` experts get (base+1) dims, the rest get base dims
    sizes = [(base + 1) if i < rem else base for i in range(K)]
    starts = [0] + list(torch.cumsum(torch.tensor(sizes), 0).tolist())

    # build Cayley Q as before
    S = A - A.t()
    I = torch.eye(dim, device=A.device, dtype=A.dtype)
    Q = torch.linalg.solve(I - S, I + S)  # (dim, dim)

    V = torch.zeros_like(U)
    for i in range(K):
        s, e = starts[i], starts[i+1]
        Bi = Q[:, s:e]           # shape (dim, sizes[i])
        ui = U[:, i]             # shape (batch, dim)
        coords = ui @ Bi         # → (batch, sizes[i])
        V[:, i] = coords @ Bi.t()# → (batch, dim)


    for i in range(K):
        M = V[:, i].T             # now shape = (dim, batch)
        rank = torch.linalg.matrix_rank(M)
        assert rank <= (dim // K + 1), (
            f"Expert {i}: empirical rank {rank} exceeds block size {dim //K}"
        )

    return V



if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()



def gram_schmidt_orthonormalise(U: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Differentiable Gram–Schmidt on U of shape (batch, K, dim).
    Avoids in-place ops by cloning and stacking.
    """
    batch, K, dim = U.shape
    orthonorms = []

    for i in range(K):
        # clone the slice so we don't modify a view of U
        v = U[:, i].clone()              # (batch, dim)

        # subtract projections onto all previous orthonormal vectors
        for vj in orthonorms:            # each vj is (batch, dim)
            # ⟨v, vj⟩ / (⟨vj, vj⟩ + eps), shape (batch,1)
            coeff = (v * vj).sum(dim=1, keepdim=True) \
                  / (vj.pow(2).sum(dim=1, keepdim=True) + eps)
            v = v - coeff * vj           # safe: v is a fresh Tensor

        # normalize to unit length
        norm = v.norm(dim=1, keepdim=True).clamp_min(eps)
        v = v / norm

        orthonorms.append(v)

    # stack back into (batch, K, dim)
    return torch.stack(orthonorms, dim=1)




def calculate_lambda_max_loss(x, batch_size, n_experts=3):   
    ''' x is shape (K, batch_size, dim) '''
    x = F.normalize(x, p=2, dim=-1)  # now normalizes each dim-vector
    x = x.permute(0, 2, 1).contiguous() 

    print(x.shape)
    eps = 1e-6

    Q, R = torch.linalg.qr(x, mode="reduced")



        
    r_diag = R.abs().diagonal(dim1=-2, dim2=-1)           # (E, min(d,B))
    k      = (r_diag > eps).sum(dim=1)   
    
    for i, ki in enumerate(k):
        print(x.shape)
        print(f"expert_{i}_empirical_rank", ki.item())
    cols   = torch.arange(Q.size(-1), device=Q.device)    # (d,)
    mask   = cols[None, None, :] < k[:, None, None]       # (E, 1, d)
    Qm     = Q * mask                                     
    projs  = Qm @ Qm.transpose(-2, -1) 
    avg_proj    = projs.mean(dim=0) 

    eigvals = torch.linalg.eigvalsh(avg_proj)
    lambda_max = eigvals[-1]
    assert R.shape[0] == n_experts
    assert R.shape[-1] == batch_size 
    print(lambda_max)
    return lambda_max.to('cuda')