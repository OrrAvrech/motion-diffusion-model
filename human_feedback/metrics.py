import torch
from torch import Tensor
from human_feedback.motion_utils import align_by_root


def compute_mpjpe(gt_joints: Tensor, pred_joints: Tensor, root_align: bool = True) -> Tensor:
    if root_align is True:
        gt_joints = align_by_root(gt_joints)
        pred_joints = align_by_root(pred_joints)
        
    joint_errors = torch.norm(gt_joints - pred_joints, dim=-1)
    mpjpe = torch.mean(joint_errors)
    return mpjpe


def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3, F).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3, F).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """

    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1, 3)
    S2 = S2.permute(0, 2, 1, 3)
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1**2).sum(dim=(1,2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0, 2, 1, 3))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1, 3)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=U.device).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1, :] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1, 3))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale*torch.matmul(R, mu1)

    # 7. Error:
    S1_hat = scale*torch.matmul(R, S1) + t

    return S1_hat.permute(0, 2, 1, 3)


def compute_pa_mpjpe(gt_joints: Tensor, pred_joints: Tensor) -> Tensor:
    # Align predicted joints to ground truth using Procrustes alignment
    pred_aligned = compute_similarity_transform(gt_joints, pred_joints)
    pampjpe = compute_mpjpe(gt_joints, pred_aligned)
    return pampjpe