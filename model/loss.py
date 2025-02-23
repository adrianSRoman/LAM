import torch
import torch.nn as nn
from trainer.utils import get_field

from scipy.spatial import KDTree

def mse_loss():
    return torch.nn.MSELoss()

def l1_loss():
    return torch.nn.L1Loss()

def laplacian_smoothing(latent_x):
    laplacian = latent_x[:, 2:] - 2 * latent_x[:, 1:-1] + latent_x[:, :-2]
    return torch.sum(laplacian**2)

class ComplexMSELoss(nn.Module):
    def __init__(self):
        super(ComplexMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, target, pred):
        # Ensure both target and pred are complex tensors
        assert target.is_complex() and pred.is_complex(), "Both tensors must be complex"

        # Separate real and imaginary parts
        target_real = target.real
        target_imag = target.imag
        pred_real = pred.real
        pred_imag = pred.imag

        # Compute MSE loss for real and imaginary parts
        mse_loss_real = self.mse_loss(target_real, pred_real)
        mse_loss_imag = self.mse_loss(target_imag, pred_imag)

        # Combine the losses
        total_loss = mse_loss_real + mse_loss_imag

        return total_loss


# L1 loss + centroid std deviation regularization
class L1RegLoss(nn.Module):
    def __init__(self, N_max=20, reg_weight=0.01, device='cuda:0'):
        super(L1RegLoss, self).__init__()
        self.R_xyz = torch.from_numpy(get_field()).to(device)
        self.N_max = N_max
        self.l1_loss = nn.L1Loss()
        self.reg_weight = reg_weight
        
    def forward(self, target, pred, latent):
        l1_loss = self.l1_loss(target, pred)
        reg_loss = self.calculate_reg_loss(latent)
        total_loss = l1_loss + self.reg_weight * reg_loss
        
        return total_loss, l1_loss, self.reg_weight * reg_loss
    
    def calculate_reg_loss(self, latent):
        max_indices = torch.topk(latent, self.N_max).indices
        max_xyz = self.R_xyz[:, max_indices].T
        
        dists = F.pdist(max_xyz, p=2)
        std_dist = torch.std(dists)
        
        return std_dist


# Mean Squared Error + Dispersion loss
class MSEDLoss(nn.Module):
    def __init__(self, N_max=20, dispersion_weight=0.01, device='cuda:0'):
        super(MSEDLoss, self).__init__()
        self.R_xyz = torch.from_numpy(get_field()).to(device)
        self.N_max = N_max
        self.mse_loss = ComplexMSELoss()
        self.dispersion_weight = dispersion_weight
        
    def forward(self, target, pred, latent):
        mse_loss = self.mse_loss(target, pred)
        dispersion_loss = self.calculate_dispersion_loss(latent)
        total_loss = mse_loss + self.dispersion_weight * dispersion_loss
        
        return total_loss, mse_loss, self.dispersion_weight * dispersion_loss
    
    def calculate_dispersion_loss(self, latent):
        max_indices = torch.topk(latent, self.N_max).indices
        max_xyz = self.R_xyz[:, max_indices].T
        
        centered_xyz = max_xyz - max_xyz.mean(dim=0)
        cov_matrix = torch.matmul(centered_xyz.T, centered_xyz) / (self.N_max - 1)
        e_vals = torch.linalg.eigvalsh(cov_matrix)
        
        return e_vals.sum()


class MSEL1Loss(nn.Module):
    def __init__(self, l1_weight=0.0001, device='cuda:0'):
        super(MSEL1Loss, self).__init__()
        self.mse_loss = ComplexMSELoss()
        self.l1_weight = l1_weight

    def forward(self, target, pred, latent):
        mse_loss = self.mse_loss(target, pred)
        #mse_loss += self.l1_weight * (torch.norm(latent, p=1) + torch.norm(latent, p=2)**2)
        #mse_loss += self.l1_weight * (torch.norm(latent, p=2)**2)
        #mse_loss += self.l1_weight * laplacian_smoothing(latent)
        return mse_loss, None, None


class MSELossWithGaussian(nn.Module):
    def __init__(self, l1_weight=0.0001, gaussian_weight=0.01, device='cuda:0'):
        super(MSELossWithGaussian, self).__init__()
        self.mse_loss = ComplexMSELoss()
        self.l2_weight = l1_weight
        self.gaussian_weight = gaussian_weight

    def forward(self, target, pred, latent_pre_activation):
        mse_loss = self.mse_loss(target, pred)
        l2_reg = self.l2_weight * torch.norm(latent_pre_activation, p=2) ** 2
        
        # Gaussian regularization
        mean_loss = torch.mean(latent_pre_activation, dim=0).pow(2).sum()
        var_loss = (torch.var(latent_pre_activation, dim=0, unbiased=False) - 1).pow(2).sum()
        gaussian_reg = self.gaussian_weight * (mean_loss + var_loss)
        
        total_loss = mse_loss + l2_reg + gaussian_reg
        return total_loss, None, None


def total_variation_loss(tensor):
    """
    Compute the Total Variation (TV) loss for a 1D tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, dim).
    
    Returns:
        torch.Tensor: The TV loss for the input tensor.
    """
    diff = tensor[:, 1:] - tensor[:, :-1]
    return torch.sum(torch.abs(diff))


def total_variation_fibonacci_loss(latent, R):
    """
    Compute the Total Variation (TV) loss for a Fibonacci grid.

    Args:
        latent (torch.Tensor): Latent tensor of shape (batch_size, N_points).
        R (numpy.ndarray): Fibonacci grid of shape (3, N_points) in Cartesian coordinates.

    Returns:
        torch.Tensor: The TV loss for the latent space.
    """
    if (len(latent.shape) == 1):
        latent = latent.unsqueeze(0)
    
    # Build a KDTree to find neighbors in the grid
    kdtree = KDTree(R.T)  # Transpose R to shape (N_points, 3)
    _, indices = kdtree.query(R.T, k=2)  # Find nearest neighbor for each point

    # Compute differences between each point and its nearest neighbor
    diff = latent[:, indices[:, 0]] - latent[:, indices[:, 1]]

    # Compute TV loss
    tv_loss = torch.sum(torch.abs(diff))

    return tv_loss

def total_variation_fibonacci_lossi_multi(latent, R, num_neighbors=6):
    """
    Compute the Total Variation (TV) loss for a Fibonacci grid considering multiple neighbors.

    Args:
        latent (torch.Tensor): Latent tensor of shape (batch_size, N_points).
        R (numpy.ndarray): Fibonacci grid of shape (3, N_points) in Cartesian coordinates.
        num_neighbors (int): Number of neighbors to consider for each point.

    Returns:
        torch.Tensor: The TV loss for the latent space.
    """
    batch_size, N_points = latent.shape

    # Build a KDTree to find neighbors in the grid
    kdtree = KDTree(R.T)  # Transpose R to shape (N_points, 3)
    _, indices = kdtree.query(R.T, k=num_neighbors + 1)  # +1 includes the point itself

    # Compute differences between each point and its neighbors
    tv_loss = 0
    for i in range(1, num_neighbors + 1):  # Skip index 0 as it is the point itself
        diff = latent[:, indices[:, 0]] - latent[:, indices[:, i]]
        tv_loss += torch.sum(torch.abs(diff))

    return tv_loss


class MSETVLoss(nn.Module):
    def __init__(self, l1_weight=0.0001, device='cuda:0', reg="l1"):
        super(MSETVLoss, self).__init__()
        self.mse_loss = ComplexMSELoss()
        self.l1_weight = l1_weight
        self.R = get_field() # Fibonacci grid of shape (3, N_points) in Cartesian coordinates.
        kdtree = KDTree(self.R.T)
        self.num_neighbors = 6
        _, indices = kdtree.query(self.R.T, k=self.num_neighbors + 1)  # +1 includes the point itself
        self.indices = indices
        self.reg = reg

    def total_variation_fibonacci_lossi(self, latent):
        """
        Compute the Total Variation (TV) loss for a Fibonacci grid considering multiple neighbors.

        Args:
            latent (torch.Tensor): Latent tensor of shape (batch_size, N_points).
        Returns:
            torch.Tensor: The TV loss for the latent space.
        """
        # Compute differences between each point and its neighbors
        tv_loss = 0
        for i in range(1, self.num_neighbors + 1):  # Skip index 0 as it is the point itself
            diff = latent[:, self.indices[:, 0]] - latent[:, self.indices[:, i]]
            tv_loss += torch.sum(torch.abs(diff))
        return tv_loss

    def forward(self, target, pred, latent):
        """
        Forward pass for the loss function.

        Args:
            target (torch.Tensor): Ground truth matrix.
            pred (torch.Tensor): Predicted matrix.
            latent (torch.Tensor): Latent vector of shape (batch_size, 484).
        
        Returns:
            torch.Tensor: Combined loss (MSE + TV loss).
        """
        mse_loss = self.mse_loss(target, pred)
        
        # Total Variation Loss for latent vector
        if self.reg == "l1": # MSE + TV + L1
            tv_loss = self.l1_weight * (self.total_variation_fibonacci_lossi(latent) + torch.norm(latent, p=1))
        elif self.reg == "l2": # MSE + TV + L2
            tv_loss = self.l1_weight * (self.total_variation_fibonacci_lossi(latent) + torch.norm(latent, p=2)**2)
        else: # MSE + TV
            tv_loss = self.l1_weight * (self.total_variation_fibonacci_lossi(latent))
        
        # Combine losses
        combined_loss = mse_loss + tv_loss

        return combined_loss, mse_loss, tv_loss

