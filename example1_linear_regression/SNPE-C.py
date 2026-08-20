#!/usr/bin/env python
# coding: utf-8

# In[17]:


"""
Implementation of the APT method with an example in Bayesian Linear Regression (Compute Canada)
Author: Renjie Peng
Date: 2025-02-19
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import the MDN from pyknos
from pyknos.mdn.mdn import MultivariateGaussianMDN
from scipy import stats
from scipy.stats import multivariate_normal

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# In[18]:


# ---- read seed & output dir from environment (set by Slurm script) ----
SEED = int(os.environ.get("SEED", 42))                # default 42 if not provided
OUTDIR = os.environ.get("OUTDIR", ".")                # default: current dir
os.makedirs(OUTDIR, exist_ok=True)
# ---- reproducibility ----
np.random.seed(SEED)
torch.manual_seed(SEED)


# In[19]:


# ---------------------------
# Prior and Simulator Functions
# ---------------------------
a0 = 6.0
b0 = 2.0
m0 = torch.tensor([1.0, 4.0], dtype=torch.float32)      # shape: (2,)
M0 = torch.tensor([[1.0, 0.0],
                   [0.0, 1.0]], dtype=torch.float32)      # shape: (2,2)
n = 5000   # number of simulated samples
m = 10       # number of design points

def sample_prior(n, m0, M0, a0, b0):
    # Sample sigma^2 from an inverse-gamma (via Gamma and inversion)
    gamma_dist = torch.distributions.Gamma(
        concentration=torch.tensor(a0 / 2),
        rate=torch.tensor(b0 / 2)
    )
    gamma_samples = gamma_dist.sample((n, 1))  # shape: (n, 1)
    sigma2 = 1.0 / gamma_samples               # shape: (n, 1)
    
    # Compute Cholesky factor of M0 (lower triangular)
    L0 = torch.linalg.cholesky(M0)             # shape: (2, 2)
    
    # Sample beta ~ N(m0, sigma2 * M0)
    z = torch.randn(n, 2, 1)                     # shape: (n, 2, 1)
    sqrt_sigma2 = torch.sqrt(sigma2)             # shape: (n, 1)
    m0_expanded = m0.unsqueeze(0).unsqueeze(2)   # shape: (1, 2, 1) -> broadcasts to (n, 2, 1)
    beta = m0_expanded + sqrt_sigma2.unsqueeze(2) * torch.matmul(L0, z)  # shape: (n,2,1)
    
    return {'beta': beta, 'sigma2': sigma2}

def simulator(beta, sigma2, m):
    n = beta.shape[0]
    # Build design matrix X: (m,2)
    X = torch.stack([torch.ones(m), torch.linspace(-10, 10, steps=m)], dim=1)
    # Sample error: (n, m, 1) with sqrt_sigma2 broadcasted over m
    sqrt_sigma2 = torch.sqrt(sigma2)   # (n, 1)
    error = torch.randn(n, m, 1) * sqrt_sigma2.unsqueeze(1)
    # Compute Y = X @ beta + error. Expand X to (n, m, 2)
    X_expanded = X.unsqueeze(0).expand(n, m, 2)   # (n, m, 2)
    Y = torch.bmm(X_expanded, beta) + error       # (n, m, 1)
    # Return Y transposed to shape: (n, 1, m)
    return Y.transpose(1, 2)

def standardize(tensor):
    # tensor: (n, features)
    mu = torch.mean(tensor, dim=0, keepdim=True)  # (1, features)
    std = torch.std(tensor, dim=0, keepdim=True)    # (1, features)
    standardized = (tensor - mu) / torch.clamp(std, min=1e-6)
    return {'mu': mu, 'std': std, 'standardized': standardized}

# ---------------------------
# Generate Training Data
# ---------------------------
prior_sim = sample_prior(n, m0, M0, a0, b0)
# Y_sim: shape (n, 1, m) → squeeze to (n, m)
Y_sim = simulator(prior_sim['beta'], prior_sim['sigma2'], m).squeeze(1)
std_res_Y = standardize(Y_sim)
Y_mean = std_res_Y['mu']  # (1, m)
Y_std = std_res_Y['std']  # (1, m)
Y_sim_stdzd = std_res_Y['standardized']  # (n, m)


# Process beta: (n,2,1) → (n,2)
beta_sim = prior_sim['beta'].transpose(1, 2).squeeze(1)
# Process log(sigma2): (n,1) → (n,)
log_sigma2_sim = torch.log(prior_sim['sigma2'].squeeze(1))
# Concatenate to form parameters: (n, 3) → [beta0, beta1, log(sigma2)]
parameters_sim = torch.cat([beta_sim, log_sigma2_sim.unsqueeze(1)], dim=1)
# Standardize the parameters
std_res = standardize(parameters_sim)
parameters_mean = std_res['mu']   # (1, 3)
parameters_std = std_res['std']     # (1, 3)
parameters_stdzd = std_res['standardized']  # (n, 3)

# ---------------------------
# Generate Observed Data
# ---------------------------
# Sample from the prior to generate parameters for the observed data
obs_prior_sample = sample_prior(1, m0, M0, a0, b0)
obs_beta = obs_prior_sample['beta']  # shape: (1, 2, 1)
obs_sigma2 = obs_prior_sample['sigma2']  # shape: (1, 1)

# Simulate observed data using the sampled parameters
Y_obs = simulator(obs_beta, obs_sigma2, m).squeeze(0).squeeze(0)  # shape: (m,)

# Standardize the observed data
Y_obs_stdzd = (Y_obs - Y_mean) / torch.clamp(Y_std, min=1e-6)
if Y_obs_stdzd.dim() == 1:
    Y_obs_stdzd = Y_obs_stdzd.unsqueeze(0)
    


# In[20]:


# ---------------------------
# Split Training and Validation Data
# ---------------------------
indices = np.random.permutation(n)
n_train = int(n * 0.9)
train_idx = indices[:n_train]
val_idx = indices[n_train:]
Y_train = Y_sim_stdzd[train_idx]          # (n_train, m)
params_train = parameters_stdzd[train_idx]  # (n_train, 3)
Y_val = Y_sim_stdzd[val_idx]              # (n_val, m)
params_val = parameters_stdzd[val_idx]  # (n_val, 3)

# ---------------------------
# Define Hidden Network for MDN
# ---------------------------
hidden_net = nn.Sequential(
    nn.Linear(m, 50),
    nn.ReLU(inplace=False),
    nn.Dropout(p=0.0),
    nn.Linear(50, 50),
    nn.ReLU(inplace=False),
    nn.Linear(50, 50),
    nn.ReLU(inplace=False)
)

# ---------------------------
# Instantiate the PyKnos MDN
# ---------------------------
# Here, 'features' is the target dimension (3) and 'context_features' is the input dimension (m)
num_components = 5
mdn = MultivariateGaussianMDN(
    features=3,
    context_features=m,
    hidden_net=hidden_net,
    num_components=num_components,  # mixture of 5 Gaussians
    custom_initialization=True
)

optimizer = optim.Adam(mdn.parameters(), lr=1e-4)

# ---------------------------
# Training Loop (using MDN's log_prob)
# ---------------------------
batch_size = 200
num_epochs = 2**31 - 1
patience_threshold = 20
num_batches = int(np.ceil(n_train / batch_size))
best_val_loss = float('inf')
patience_counter = 0
best_state = None
clip_max_norm = 5.0  # maximum gradient norm

for epoch in range(1, num_epochs + 1):
    mdn.train()
    total_loss = 0.0
    permutation = np.random.permutation(n_train)
    for i in range(0, n_train, batch_size):
        optimizer.zero_grad()
        batch_idx = permutation[i:i+batch_size]
        Y_batch = Y_train[batch_idx]           # context: (batch, m)
        target_batch = params_train[batch_idx]   # target: (batch, 3)
        # Compute log probability of target given context
        log_prob = mdn.log_prob(target_batch, Y_batch)
        loss = -log_prob.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mdn.parameters(), max_norm=clip_max_norm)
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / num_batches

    mdn.eval()
    with torch.no_grad():
        val_log_prob = mdn.log_prob(params_val, Y_val)
        val_loss = -val_log_prob.mean().item()
    
    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = mdn.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= patience_threshold:
        print(f"Early stopping at epoch {epoch}.")
        break

if best_state is not None:
    mdn.load_state_dict(best_state)
    print("Loaded best MDN weights.")


# In[21]:


# ---------------------------
# Get Mean and Covariance of Mixture Components
# ---------------------------
logits, means, precisions, _, _  = mdn.get_mixture_components(Y_obs_stdzd)
covs = torch.linalg.inv(precisions)
D = torch.diag(parameters_std.squeeze())
D_expanded = D.unsqueeze(0).unsqueeze(0)     # now shape (1, 1, 3, 3)

mean_unstd = parameters_mean + parameters_std * means
cov_unstd = D_expanded @ covs @ D_expanded

mean_list   = mean_unstd.detach().cpu().numpy()   # [m1, m2, m3]
cov_matrix  = cov_unstd.detach().cpu().numpy()




# In[22]:


##########################
#Sample from the estimated posterior
##########################
mdn.eval()
with torch.no_grad():
    # Sample 10,000 samples from the estimated posterior for the observed Y_obs_stdzd
    samples = mdn.sample(n, Y_obs_stdzd)  # shape: (batch, n, 3)
# For a single observed context, squeeze the batch dimension: (n, 3)
samples = samples.squeeze(0)

# Unstandardize: original = std * sample + mean
samples_unstd = samples * parameters_std + parameters_mean  # broadcasting over (n, 3)


# In[23]:


########################## 
#Sample from the Correct Posterior
# ##########################
    
# Reconstruct the design matrix X.
X = torch.stack([
    torch.ones(m),
    torch.linspace(-10, 10, m)
], dim=1)

M0_inv = torch.inverse(M0)
M_n = torch.inverse(M0_inv + X.t() @ X)

m_n = M_n @ (M0_inv @ m0 + X.t() @ Y_obs)
c_n = (Y_obs @ Y_obs) + (m0 @ (M0_inv @ m0)) - (m_n @ (torch.inverse(M_n) @ m_n))
a_n = a0 + m
b_n = b0 + c_n

correct_sigma2_dist = torch.distributions.InverseGamma(a_n/2, b_n/2)

sigma2_samples = correct_sigma2_dist.sample((n,))

correct_beta_samples = []

for i in range(n):
    cov = sigma2_samples[i] * M_n
    mvn = torch.distributions.MultivariateNormal(m_n, covariance_matrix=cov)
    correct_beta_samples.append(mvn.sample())

correct_beta_samples = torch.stack(correct_beta_samples, dim=0)
correct_log_sigma2 = torch.log(sigma2_samples).unsqueeze(1)
correct_samples = torch.cat([correct_beta_samples, correct_log_sigma2], dim=1)




# In[24]:


#save mean_list and cov_list for round 1
logits, means, precisions, _, _  = mdn.get_mixture_components(Y_obs_stdzd)
covs = torch.linalg.inv(precisions)
D = torch.diag(parameters_std.squeeze())
D_expanded = D.unsqueeze(0).unsqueeze(0)     # now shape (1, 1, 3, 3) for broadcasting
mean_unstd = parameters_mean + parameters_std * means # shape: (1, num_components, 3)
cov_unstd = D_expanded @ covs @ D_expanded  # shape: (1, num_components, 3, 3)

# Convert to numpy
means_np = mean_unstd[0].detach().cpu().numpy()  # (num_components, 3)
covs_np = cov_unstd[0].detach().cpu().numpy()    # (num_components, 3, 3)
weights_np = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()  # (num_components,)

# Save as .npz file (compressed numpy format)
np.savez_compressed(
    os.path.join(OUTDIR, f"mixture_params_round1_seed{SEED}.npz"),
    means=means_np,
    covariances=covs_np,
    weights=weights_np,
    num_components=num_components
)

print(f"Saved mixture parameters for round 1 as .npz file")


# In[25]:


##########################
#Compare Marginal Distributions and Save Plot
##########################
# Convert samples to numpy arrays.
est_samples_np = samples_unstd.detach().cpu().numpy()
corr_samples_np = correct_samples.detach().cpu().numpy()
param_names = ['beta0', 'beta1', 'log(sigma2)']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, param in enumerate(param_names):
    sns.kdeplot(corr_samples_np[:, i], ax=axes[i], label='Correct', color='blue')
    sns.kdeplot(est_samples_np[:, i], ax=axes[i], label='Estimated', color='red')
    axes[i].set_title(f"Marginal for {param}")
    axes[i].legend()

plt.tight_layout()

plot_filename = os.path.join(OUTDIR, f"comparison_plot_round_{1}_seed{SEED}.png")

plt.savefig(plot_filename)
print(f"Saved comparison plot as {plot_filename}")


# est_samples_np has shape (n_samples, 3)
np.savetxt(
    os.path.join(OUTDIR, f"posterior_estimated1_seed{SEED}.csv"),
    est_samples_np,
    delimiter=",",
    header="beta0,beta1,log(sigma2)",
    comments=""      # prevents NumPy from prefixing the header with '#'
)

#Classifier 2-Sample Tests (C2ST)
acc_per_round = []

# Stack samples and labels
X = np.vstack([corr_samples_np, est_samples_np])
y = np.concatenate([np.zeros(len(corr_samples_np)), np.ones(len(est_samples_np))])

# Shuffle
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train classifier
clf = LogisticRegression(max_iter=1000, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
acc_per_round.append(acc)
print(f"Classifier accuracy (first round): {acc:.2f}")

np.savetxt(
        os.path.join(OUTDIR, f"acc_per_round_seed{SEED}.csv"),
        acc_per_round,
        delimiter=",",
        header="Accuracy",
        comments=""  # Prevents NumPy from adding a '#' before the header
    )


# In[26]:


# Likelihood Ratio Test (LRT) - Density-based classification
acc_den_per_round = []

# (M_n, m_n, a_n, b_n are already available from the correct posterior sampling section)
def compute_correct_posterior_logdensity(theta_samples):
    """Compute density of correct posterior at given theta samples"""
    # Convert to tensor only if needed for mathematical operations
    if isinstance(theta_samples, np.ndarray):
        theta_tensor = torch.tensor(theta_samples, dtype=torch.float32)
    else:
        theta_tensor = theta_samples

    a_n_tensor = torch.tensor(a_n, dtype=torch.float32)
    b_n_tensor = torch.tensor(b_n, dtype=torch.float32)
    
    beta = theta_tensor[:, :2]  # (n_samples, 2)
    log_sigma2 = theta_tensor[:, 2]  # (n_samples,)
    sigma2 = torch.exp(log_sigma2)

    log_p_sigma2 = (a_n_tensor/2) * torch.log(b_n_tensor/2) - torch.lgamma(a_n_tensor/2) - ((a_n_tensor/2) + 1) * torch.log(sigma2) - b_n_tensor/(2*sigma2)

    diff = beta - m_n.unsqueeze(0)  # (n_samples, 2)
    M_n_inv = torch.inverse(M_n)
    quad = torch.sum(diff * (diff @ M_n_inv.t()), dim=1) / sigma2
    log_p_beta = -torch.log(torch.tensor(2 * torch.pi)) - 0.5 * torch.log(torch.det(M_n)) - torch.log(sigma2) - 0.5 * quad

    log_joint = log_p_sigma2 + log_p_beta + log_sigma2
    return log_joint

def compute_estimated_posterior_logdensity(theta_samples):
    """Compute density of estimated posterior at given theta samples"""
    # Convert to tensor for standardization and MDN operations
    if isinstance(theta_samples, np.ndarray):
        theta_tensor = torch.tensor(theta_samples, dtype=torch.float32)
    else:
        theta_tensor = theta_samples
    
    # Standardize samples
    theta_std = (theta_tensor - parameters_mean) / parameters_std
    Y_expanded = Y_obs_stdzd.expand(len(theta_tensor), -1)
    
    with torch.no_grad():
        log_probs_std = mdn.log_prob(theta_std, Y_expanded)

    # Transform density from standardized space to original space
    # p(theta) = p(theta_std) * |J| where J is the Jacobian of the transformation
    # For linear transformation theta_std = (theta - mu) / sigma:
    # |J| = 1 / prod(sigma) = 1 / prod(parameters_std)
    log_jacobian = -torch.sum(torch.log(parameters_std), dim=-1)  # log(1/prod(std))
    
    # Density in original space
    log_probs_original = log_probs_std + log_jacobian
    return log_probs_original


def density_based_classification(corr_samples_np, est_samples_np, round_num=1, verbose=True):
    """
    Perform density-based classification between correct and estimated posterior samples.
    
    Args:
        corr_samples_np: numpy array of correct posterior samples (n_samples, n_params)
        est_samples_np: numpy array of estimated posterior samples (n_samples, n_params)
        round_num: int, round number for logging
        verbose: bool, whether to print results
    
    Returns:
        dict: {
            'accuracy': float,
            'zero_diff_count': int,
            'zero_diff_percentage': float,
            'total_samples': int
        }
    """
    # Stack all samples and create labels
    all_samples = np.vstack([corr_samples_np, est_samples_np])
    true_labels = np.concatenate([np.zeros(len(corr_samples_np)), np.ones(len(est_samples_np))])


    # Compute LOG densities for all samples 
    log_p_correct = compute_correct_posterior_logdensity(all_samples)
    log_p_estimated = compute_estimated_posterior_logdensity(all_samples)

    # Compute log likelihood differences
    log_likelihood_differences = log_p_correct - log_p_estimated

    # Count samples with difference = 0
    zero_diff_count = (log_likelihood_differences == 0).sum().item()
    total_samples = len(log_likelihood_differences)
    zero_diff_percentage = (zero_diff_count / total_samples) * 100

    if verbose:
        print(f"Round {round_num} - Samples with log likelihood difference = 0: {zero_diff_count} / {total_samples} ({zero_diff_percentage:.2f}%)")

    # Classify based on likelihood difference with random assignment for ties
    predicted_labels = torch.zeros_like(log_likelihood_differences)

    # Clear cases: p_correct > p_estimated → label 0 (correct posterior)
    predicted_labels[log_likelihood_differences > 0] = 0

    # Clear cases: p_correct < p_estimated → label 1 (estimated posterior)  
    predicted_labels[log_likelihood_differences < 0] = 1

    # Tie cases: p_correct = p_estimated → random assignment
    zero_mask = (log_likelihood_differences == 0)
    if zero_mask.sum() > 0:
        random_labels = torch.randint(0, 2, (zero_mask.sum(),), dtype=torch.float32)
        predicted_labels[zero_mask] = random_labels

    predicted_labels = predicted_labels.numpy()

    # Calculate accuracy
    accuracy_den = (predicted_labels == true_labels).mean()

    if verbose:
        print(f"Round {round_num} - Density difference accuracy: {accuracy_den:.2f}")

    return {
        'accuracy': accuracy_den,
        'zero_diff_count': zero_diff_count,
        'zero_diff_percentage': zero_diff_percentage,
        'total_samples': total_samples
    }

# Usage in your existing code:

# For the first round (replace your existing density classification code):
density_results = density_based_classification(corr_samples_np, est_samples_np, round_num=1)
acc_den_per_round.append(density_results['accuracy'])
# Save density-based accuracy
np.savetxt(
    os.path.join(OUTDIR, f"acc_den_per_round_seed{SEED}.csv"),
    acc_den_per_round,
    delimiter=",",
    header="Density_Accuracy",
    comments=""
)


# In[27]:


# monte carlo estimation of KL divergence
kl_divergence_per_round = []
def compute_empirical_kl_divergence_mc(corr_samples_np, verbose=True):
    """
    Compute empirical KL divergence D_KL(True||Estimated) using Monte Carlo estimation.
    
    KL(P||Q) = E_P[log(p(x)/q(x))] ≈ (1/N) * Σ log(p(x_i)/q(x_i))
    where x_i are samples from the TRUE posterior P.
    
    Args:
        corr_samples_np: numpy array - samples from TRUE posterior (reference)
        verbose: bool - whether to print debug info
        
    Returns:
        float - empirical KL divergence estimate D_KL(True||Estimated)
    """
    n_samples = len(corr_samples_np)
    
    # Convert to torch tensors for density computation
    corr_samples_torch = torch.tensor(corr_samples_np, dtype=torch.float32)
    
    # Compute true posterior density at samples from true posterior
    log_p_true = compute_correct_posterior_logdensity(corr_samples_torch)
    
    # Compute estimated posterior density at samples from true posterior
    log_q_est = compute_estimated_posterior_logdensity(corr_samples_torch)

    # Support mismatch -> KL = +∞
    neg_inf = torch.isneginf(log_q_est)
    if torch.any(neg_inf):
        frac = neg_inf.double().mean().item()
        if verbose:
            print(f"[KL] q(θ) = 0 on {100*frac:.2f}% of P-samples ⇒ KL = +∞.")
        return float('inf')
    
    # Compute log ratio: log(p/q)
    log_ratios = log_p_true - log_q_est

    # Monte Carlo estimate: KL ≈ (1/N) * Σ log(p(x_i)/q(x_i))
    kl_divergence = torch.mean(log_ratios).item()
    
    if verbose:
        print(f"  MC-KL: mean log(p/q) = {kl_divergence:.4f}")
        print(f"  MC-KL: min log(p/q) = {torch.min(log_ratios).item():.4f}")
        print(f"  MC-KL: max log(p/q) = {torch.max(log_ratios).item():.4f}")
    
    return kl_divergence
kl_div = compute_empirical_kl_divergence_mc(corr_samples_np, verbose=True)
kl_divergence_per_round.append(kl_div)
# Save KL divergence
np.savetxt(
    os.path.join(OUTDIR, f"kl_divergence_per_round_seed{SEED}.csv"),
    kl_divergence_per_round,
    delimiter=",",
    header="KL_Divergence_MC",
    comments=""
)


# In[28]:


# ---------------------------
# Exact Prior in Standardized Space
# ---------------------------
class ExactPriorStandardized(torch.distributions.Distribution):
    def __init__(self, parameters_mean, parameters_std, m0, M0, a0, b0):
        super().__init__()
        self.parameters_mean = parameters_mean  # (1, 3)
        self.parameters_std = parameters_std  # (1, 3)
        self.m0 = m0
        self.M0 = M0
        self.a0 = torch.tensor(a0, dtype=torch.float32, device=parameters_mean.device)
        self.b0 = torch.tensor(b0, dtype=torch.float32, device=parameters_mean.device)
        self.dim = 3

    def log_prob(self, theta_stdzd):
        # Transform back to original space:
        theta = theta_stdzd * self.parameters_std + self.parameters_mean
        beta = theta[..., :2]
        log_sigma2 = theta[..., 2]
        sigma2 = torch.exp(log_sigma2)

        # Inverse Gamma density for sigma2:
        log_p_sigma2 = (self.a0 / 2) * torch.log(self.b0 / 2) \
                       - torch.lgamma(self.a0 / 2) \
                       - ((self.a0 / 2) + 1) * torch.log(sigma2) \
                       - self.b0 / (2 * sigma2)

        # Conditional for beta | sigma2:
        d = 2
        M0_inv = torch.inverse(self.M0)
        diff = beta - self.m0
        quad = torch.sum(diff * (diff @ M0_inv), dim=-1)
        log_det_M0 = torch.log(torch.det(self.M0))
        log_p_beta_given_sigma2 = -d / 2 * torch.log(
            2 * torch.tensor(np.pi, dtype=theta.dtype, device=theta.device) * sigma2
        ) - 0.5 * log_det_M0 - quad / (2 * sigma2)

        # Jacobian for sigma2 -> log_sigma2:
        log_joint = log_p_sigma2 + log_p_beta_given_sigma2 + log_sigma2

        # Adjustment for standardization transform:
        log_det = torch.sum(torch.log(self.parameters_std), dim=-1)
        return log_joint + log_det

    def sample(self, sample_shape=torch.Size()):
        n = int(np.prod(sample_shape)) if sample_shape != torch.Size() else 1
        raw_samples = sample_prior(n, m0=self.m0, M0=self.M0,
                                   a0=self.a0.item(), b0=self.b0.item())
        beta = raw_samples['beta'].squeeze(-1)  # (n, 2)
        log_sigma2 = torch.log(raw_samples['sigma2'].squeeze(1))  # (n,)
        theta = torch.cat([beta, log_sigma2.unsqueeze(1)], dim=1)  # (n, 3)
        theta_stdzd = (theta - self.parameters_mean) / self.parameters_std
        return theta_stdzd

    def p_0(self, theta_unstandardized):

        beta, log_sigma2 = theta_unstandardized[:, :2], theta_unstandardized[:, 2]

        sigma2 = torch.exp(log_sigma2)

        # Inverse Gamma density for sigma2:
        log_p_sigma2 = (self.a0 / 2) * torch.log(self.b0 / 2) \
                       - torch.lgamma(self.a0 / 2) \
                       - ((self.a0 / 2) + 1) * torch.log(sigma2) \
                       - self.b0 / (2 * sigma2)

        # Conditional for beta | sigma2:
        d = 2
        M0_inv = torch.inverse(self.M0)
        diff = beta - self.m0
        quad = torch.sum(diff * (diff @ M0_inv), dim=-1)
        log_det_M0 = torch.log(torch.det(self.M0))
        log_p_beta_given_sigma2 = -d / 2 * torch.log(
            2 * torch.tensor(np.pi, dtype=theta_unstandardized.dtype, device=theta_unstandardized.device) * sigma2
        ) - 0.5 * log_det_M0 - quad / (2 * sigma2)

        # Jacobian for sigma2 -> log_sigma2:
        log_joint = log_p_sigma2 + log_p_beta_given_sigma2 + log_sigma2

        # Adjustment for standardization transform:
        log_det = torch.sum(torch.log(self.parameters_std), dim=-1)
        return torch.exp(log_joint + log_det)

        



# In[30]:


# Held-out data set for posterior selection
prior = ExactPriorStandardized(parameters_mean, parameters_std, m0, M0, a0, b0)
k = 100  # top-k high-posterior parameters
n_proposal = n

# Step 1: Identify top-k high-posterior parameters
posterior_samples_std = mdn.sample(n_proposal, Y_obs_stdzd).squeeze(0)
log_probs_post = mdn.log_prob(posterior_samples_std, Y_obs_stdzd.expand(n_proposal, -1))
probs_post = torch.exp(log_probs_post)

topk_idx = torch.topk(log_probs_post, k).indices
theta_star_std_list = posterior_samples_std[topk_idx]
w_list = probs_post[topk_idx]

# Step 2: Generate held-out test data
test_samples_std = prior.sample((n_proposal*5,))
test_samples = test_samples_std * parameters_std + parameters_mean
beta_test = test_samples[:, :2]
sigma2_test = torch.exp(test_samples[:, 2]).unsqueeze(1)
Y_test = simulator(beta_test.unsqueeze(2), sigma2_test, m).squeeze(1)
Y_test_std = (Y_test - Y_mean) / Y_std
            
log_prob_matrix_test = torch.stack([
    mdn.log_prob(theta_star_std_list[i].unsqueeze(0).expand(Y_test_std.size(0), -1), Y_test_std)
    for i in range(k)
    ], dim=1)
weighted_logprob_test = (w_list * log_prob_matrix_test).sum(dim=1)

selected_idx = torch.topk(weighted_logprob_test,k).indices
theta_heldout_std = test_samples_std[selected_idx]
Y_heldout_std = Y_test_std[selected_idx]
print("Completed parameter selection on held-out data.")


# In[31]:


##########################
# EVALUATION FUNCTION FOR HELD-OUT SET
##########################
heldout_scores_per_round = []

# Evaluate Round 1
with torch.no_grad():
    heldout_score_r1 = -mdn.log_prob(theta_heldout_std, 
                                    Y_obs_stdzd.expand(theta_heldout_std.shape[0], -1)).mean().item()
    heldout_scores_per_round.append(heldout_score_r1)
    print(f"Round 1 - Held-out NLL: {heldout_score_r1:.4f}")

# Save
np.savetxt(
    os.path.join(OUTDIR, f"KL_based_score{SEED}.csv"),
    heldout_scores_per_round,
    delimiter=",",
    header="Heldout_NLL",
    comments=""
)

########################
# negative log likelihood 
########################
nll_per_round = []

# Evaluate Round 1
with torch.no_grad():
    nll_score_r1 = -mdn.log_prob(theta_heldout_std, 
                                    Y_heldout_std).mean().item()
    nll_per_round.append(nll_score_r1)
    print(f"Round 1 - negative log likelihood: {nll_score_r1:.4f}")

# Save
np.savetxt(
    os.path.join(OUTDIR, f"nll_scores{SEED}.csv"),
    nll_per_round,
    delimiter=",",
    header="nll",
    comments=""
)



# In[32]:


# Define atomic APT loss from sbi
def apt_loss_batch(mdn, thetas_batch, Y_batch, prior_std, num_atoms=10):
    B = thetas_batch.size(0)
    device = thetas_batch.device
    loss_vec = []

    for i in range(B):
        x_i = Y_batch[i].unsqueeze(0)
        theta_i = thetas_batch[i].unsqueeze(0)

        # Sample a batch of candidates including i exactly once
        all_indices = torch.arange(B)
        other_indices = all_indices[all_indices != i]
        chosen = other_indices[torch.randperm(B - 1)[:num_atoms - 1]]
        candidate_indices = torch.cat([chosen, torch.tensor([i], device=thetas_batch.device)])

        theta_candidates = thetas_batch[candidate_indices]
        log_p0 = prior_std.log_prob(theta_candidates)
        x_expanded = x_i.expand(num_atoms, -1)
        log_q = mdn.log_prob(theta_candidates, x_expanded)

        score = log_q - log_p0
        diag_index = (candidate_indices == i).nonzero(as_tuple=True)[0].item()
        loss_i = -score[diag_index] + torch.logsumexp(score, dim=0)
        loss_vec.append(loss_i)


    return torch.stack(loss_vec).mean()



# In[33]:


# ---------------------------
# Accumulate first-round data
# ---------------------------
all_theta = [parameters_sim]
all_Y = [Y_sim]


# In[34]:


for rnd in range(2,31):

    optimizer = torch.optim.Adam(mdn.parameters(), lr=5e-4)  #Re-init optimizer

    print(f"\n=== Round {rnd} ===")
    with torch.no_grad():
        samples = mdn.sample(n, Y_obs_stdzd).squeeze(0)
        samples = samples.detach()
    # Unstandardize: original = std * sample + mean
    samples_unstd = samples * parameters_std + parameters_mean  # broadcasting over (n, 3)

    beta, log_sigma2 = samples_unstd[:, :2], samples_unstd[:, 2]
    sigma2 = torch.exp(log_sigma2).unsqueeze(1)
    with torch.no_grad():
        Y_sim_r = simulator(beta.unsqueeze(2), sigma2, m).squeeze(1)
        Y_sim_r = Y_sim_r.detach()

    # Accumulate and restandardize
    all_theta.append(samples_unstd)
    all_Y.append(Y_sim_r)
    theta_all = torch.cat(all_theta, dim=0)
    Y_all = torch.cat(all_Y, dim=0)



    Y_all_std     = (Y_all - Y_mean ) / Y_std
    theta_all_std = (theta_all - parameters_mean) / parameters_std


    # Split
    n_total = theta_all_std.shape[0]
    idx = torch.randperm(n_total)
    theta_train, theta_val = theta_all_std[idx[:int(0.9*n_total)]], theta_all_std[idx[int(0.9*n_total):]]
    Y_train, Y_val = Y_all_std[idx[:int(0.9*n_total)]], Y_all_std[idx[int(0.9*n_total):]]

    # Train a new MDN with APT
    #mdn = MultivariateGaussianMDN(features=3, context_features=m, hidden_net=hidden_net, num_components=1, custom_initialization=True)

    # load in the previous round best weights
    #mdn.load_state_dict(best_state)

    #optimizer = optim.Adam(mdn.parameters(), lr=1e-4)

    # Fix: re-initialize best_state and patience_counter for each round
    best_state = None
    patience_counter = 0
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        mdn.train()
        perm = torch.randperm(len(theta_train))
        epoch_loss = 0.0
        for i in range(0, len(theta_train), batch_size):
            idx = perm[i:i+batch_size]
            batch_theta, batch_Y = theta_train[idx], Y_train[idx]

            optimizer.zero_grad()
            # Use updated mean/std for prior_std
            loss = apt_loss_batch(
                mdn,
                batch_theta,
                batch_Y,
                prior_std=ExactPriorStandardized(parameters_mean, parameters_std, m0, M0, a0, b0),
                num_atoms=10
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdn.parameters(), max_norm=clip_max_norm)
            optimizer.step()
            epoch_loss += loss.item()

        mdn.eval()
        with torch.no_grad():
            val_loss = apt_loss_batch(
                mdn,
                theta_val,
                Y_val,
                prior_std=ExactPriorStandardized(parameters_mean, parameters_std, m0, M0, a0, b0),
                num_atoms=20
            ).item()
        print(f"Epoch {epoch+1}: Train Loss = {epoch_loss:.4f}, Val Loss = {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = mdn.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_threshold:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        mdn.load_state_dict(best_state)
        print("✅ Loaded best MDN for Round", rnd)
        # Sample from the final MDN (standardized space)

    posterior_samples_std = mdn.sample(n, Y_obs_stdzd).squeeze(0)

    # Unstandardize to original parameter space
    posterior_samples = posterior_samples_std * parameters_std + parameters_mean

    # Detach and convert to NumPy
    posterior_np = posterior_samples.detach().cpu().numpy()

    # Plot marginals
    param_names = ['beta0', 'beta1', 'log(sigma2)']
    plt.figure(figsize=(15, 4))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        sns.kdeplot(corr_samples_np[:, i], label="Correct", color="blue")
        sns.kdeplot(posterior_np[:, i], label="Estimated", color="red")
        plt.title(f"Marginal for {param_names[i]}")
        plt.legend()
    plt.tight_layout()
    # plt.show()
    # Save the plot for this round.
    plot_filename =  os.path.join(OUTDIR, f"comparison_plot_round_{rnd}_seed{SEED}.png")
    plt.savefig(plot_filename)
    print(f"Saved comparison plot as {plot_filename}")

    #C2ST
    X = np.vstack([corr_samples_np, posterior_np])
    y = np.concatenate([np.zeros(len(corr_samples_np)), np.ones(len(posterior_np))])
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    acc_per_round.append(acc)
    print(f"Classifier accuracy (round {rnd}): {acc:.2f}")

    #save acc_per_round
    np.savetxt(
        os.path.join(OUTDIR, f"acc_per_round_seed{SEED}.csv"),
        acc_per_round,
        delimiter=",",
        header="Accuracy",
        comments=""  # Prevents NumPy from adding a '#' before the header
    )

    # density based classification:
    density_results = density_based_classification(corr_samples_np, posterior_np, round_num=rnd)
    acc_den_per_round.append(density_results['accuracy'])
    
    # Save density-based accuracy after each round
    np.savetxt(
        os.path.join(OUTDIR, f"acc_den_per_round_seed{SEED}.csv"),
        acc_den_per_round,
        delimiter=",",
        header="Density_Accuracy",
        comments=""
    )

    # KL divergence
    kl_div = compute_empirical_kl_divergence_mc(corr_samples_np, verbose=True)
    kl_divergence_per_round.append(kl_div)
    # Save KL divergence
    np.savetxt(
        os.path.join(OUTDIR, f"kl_divergence_per_round_seed{SEED}.csv"),
        kl_divergence_per_round,
        delimiter=",",
        header="KL_Divergence_MC",
        comments=""
    )

    #save the data simulated
    filename = os.path.join(OUTDIR, f"posterior_estimated{rnd}_seed{SEED}.csv")
    np.savetxt(
        filename,
        posterior_np,
        delimiter=",",
        header="beta0,beta1,log(sigma2)",
        comments=""
    )

    # EVALUATE ON HELD-OUT SET (against x_obs)
    with torch.no_grad():
        heldout_score = -mdn.log_prob(theta_heldout_std, Y_obs_stdzd.expand(theta_heldout_std.shape[0], -1)).mean().item()
        heldout_scores_per_round.append(heldout_score)
        print(f"Round {rnd} - Held-out NLL: {heldout_score:.4f}")
    
    # Save after each round
    np.savetxt(
        os.path.join(OUTDIR, f"KL_based_score{SEED}.csv"),
        heldout_scores_per_round,
        delimiter=",",
        header="Heldout_NLL",
        comments=""
    )

    # NLL on held-out data
    with torch.no_grad():
        nll_score = -mdn.log_prob(theta_heldout_std, 
                                        Y_heldout_std).mean().item()
        nll_per_round.append(nll_score)
        print(f"Round {rnd} - negative log likelihood: {nll_score:.4f}")

    # Save NLL after each round
    np.savetxt(
        os.path.join(OUTDIR, f"nll_scores{SEED}.csv"),
        nll_per_round,
        delimiter=",",
        header="nll",
        comments=""
    )

    #save parameters of mdn fitted

    logits, means, precisions, _, _  = mdn.get_mixture_components(Y_obs_stdzd)
    covs = torch.linalg.inv(precisions)
    D = torch.diag(parameters_std.squeeze())
    D_expanded = D.unsqueeze(0).unsqueeze(0)     # now shape (1, 1, 3, 3) for broadcasting
    mean_unstd = parameters_mean + parameters_std * means # shape: (1, num_components, 3)
    cov_unstd = D_expanded @ covs @ D_expanded  # shape: (1, num_components, 3, 3)

    # Convert to numpy
    means_np = mean_unstd[0].detach().cpu().numpy()  # (num_components, 3)
    covs_np = cov_unstd[0].detach().cpu().numpy()    # (num_components, 3, 3)
    weights_np = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()  # (num_components,)

    # Save as .npz file (compressed numpy format)
    np.savez_compressed(
        os.path.join(OUTDIR, f"mixture_params_round{rnd}_seed{SEED}.npz"),
        means=means_np,
        covariances=covs_np,
        weights=weights_np,
        num_components=num_components
    )

    print(f"Saved mixture parameters for round {rnd} as .npz file")





# In[36]:


# After all rounds, plot accuracy
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(acc_per_round) + 1), acc_per_round, marker='o')
plt.xlabel("Round")
plt.ylabel("Classifier Accuracy")
plt.title("Classifier Accuracy per Round")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, f"accuracy_per_round_seed{SEED}.png"))
print("Saved accuracy plot as accuracy_per_round.png")

