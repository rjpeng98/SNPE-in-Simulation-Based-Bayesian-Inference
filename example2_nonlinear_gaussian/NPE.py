#!/usr/bin/env python
# coding: utf-8

# In[70]:


"""
Example 2 (nonlinear regression, Gaussian errors) with NPE
Author: Renjie Peng
Date: 2026-1-19
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
from scipy.special import gammaln

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from cmdstanpy import CmdStanModel
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent



# In[71]:


# ---- read seed & output dir from environment (set by Slurm script) ----
SEED = int(os.environ.get("SEED", 42))                # default 42 if not provided
OUTDIR = os.environ.get("OUTDIR", ".")                # default: current dir
os.makedirs(OUTDIR, exist_ok=True)
# ---- reproducibility ----
np.random.seed(SEED)
torch.manual_seed(SEED)


# In[72]:


# ---------------------------
# Prior and Simulator Functions for New Model
# ---------------------------
# Model: Y(X) = θ₁e^(-θ₂X) + θ₃e^(-(θ₃+θ₄)X) + ε, where ε ~ N(0, σ²)
# Prior: θᵢ ~ U(0, 10) for i = 1,2,3,4; σ² ~ InvGamma(2, 1)

n = 5000   # number of simulated samples per round
m = 10   # number of design points (equally spaced in [0, 10])

def sample_prior(n):
    """Sample from prior: θᵢ ~ U(0, 10) for i = 1,2,3,4; σ² ~ InvGamma(2, 1)"""
    theta = torch.rand(n, 4) * 10.0  # shape: (n, 4)
    # Sample σ² from InvGamma(2, 1)
    # InvGamma(a, b) can be sampled as 1/Gamma(a, b)
    sigma2 = 1.0 / torch.distributions.Gamma(2.0, 1.0).sample((n, 1))
    # Concatenate: [θ₁, θ₂, θ₃, θ₄, σ²]
    params = torch.cat([theta, sigma2], dim=1)  # shape: (n, 5)
    return params

def simulator(params, m):
    """
    Simulate data Y given parameters.
    Model: Y(X) = θ₁e^(-θ₂X) + θ₃e^(-(θ₃+θ₄)X) + ε, where ε ~ N(0, σ²)
    
    Args:
        params: (n, 5) tensor of parameters [θ₁, θ₂, θ₃, θ₄, σ²]
        m: number of design points
    
    Returns:
        Y: (n, m) tensor of simulated observations
    """
    n = params.shape[0]
    # Design points: equally spaced in [0, 10]
    X = torch.linspace(0, 10, steps=m)  # shape: (m,)
    X_expanded = X.unsqueeze(0).expand(n, m)  # shape: (n, m)
    
    # Extract parameters
    theta1 = params[:, 0].unsqueeze(1)  # (n, 1)
    theta2 = params[:, 1].unsqueeze(1)  # (n, 1)
    theta3 = params[:, 2].unsqueeze(1)  # (n, 1)
    theta4 = params[:, 3].unsqueeze(1)  # (n, 1)
    sigma2 = params[:, 4].unsqueeze(1)  # (n, 1)
    
    # Compute deterministic part: θ₁e^(-θ₂X) + θ₃e^(-(θ₃+θ₄)X)
    term1 = theta1 * torch.exp(-theta2 * X_expanded)
    term2 = theta3 * torch.exp(-(theta3 + theta4) * X_expanded)
    
    # Add Gaussian noise ε ~ N(0, σ²)
    sigma = torch.sqrt(sigma2)
    epsilon = torch.randn(n, m) * sigma
    
    Y = term1 + term2 + epsilon  # shape: (n, m)
    return Y

def standardize(tensor):
    # tensor: (n, features)
    mu = torch.mean(tensor, dim=0, keepdim=True)  # (1, features)
    std = torch.std(tensor, dim=0, keepdim=True)    # (1, features)
    standardized = (tensor - mu) / torch.clamp(std, min=1e-6)
    return {'mu': mu, 'std': std, 'standardized': standardized}


# In[73]:


# ---------------------------
# Transformation Functions for Bounded Parameters
# ---------------------------
def transform_to_unconstrained(params, lower=0.0, upper=10.0):
    """
    Transform parameters to unconstrained space.
    - First 4 parameters (θ): logit transformation for [lower, upper]
    - Last parameter (σ²): log transformation for (0, ∞)
    
    Args:
        params: (n, 5) tensor [θ₁, θ₂, θ₃, θ₄, σ²]
    """
    theta = params[:, :4]
    sigma2 = params[:, 4:5]
    
    # Logit transform for θ ∈ [lower, upper]
    theta_normalized = (theta - lower) / (upper - lower)
    theta_normalized = torch.clamp(theta_normalized, 1e-6, 1 - 1e-6)
    theta_unconstrained = torch.log(theta_normalized / (1 - theta_normalized))
    
    # Log transform for σ² ∈ (0, ∞)
    sigma2_unconstrained = torch.log(sigma2)
    
    return torch.cat([theta_unconstrained, sigma2_unconstrained], dim=1)

def transform_to_constrained(params_unconstrained, lower=0.0, upper=10.0):
    """
    Transform unconstrained parameters back to original space.
    - First 4 parameters: inverse logit to [lower, upper]
    - Last parameter: exp to (0, ∞)
    
    Args:
        params_unconstrained: (n, 5) tensor in unconstrained space
    """
    theta_unconstrained = params_unconstrained[:, :4]
    sigma2_unconstrained = params_unconstrained[:, 4:5]
    
    # Inverse logit for θ
    theta_normalized = torch.sigmoid(theta_unconstrained)
    theta = lower + (upper - lower) * theta_normalized
    
    # Exp for σ²
    sigma2 = torch.exp(sigma2_unconstrained)
    
    return torch.cat([theta, sigma2], dim=1)


# In[74]:


# ---------------------------
# Generate Training Data
# ---------------------------
theta_sim = sample_prior(n)  # shape: (n, 5) now includes σ²
Y_sim = simulator(theta_sim, m)  # shape: (n, m)

# Transform parameters to unconstrained space
theta_sim_unconstrained = transform_to_unconstrained(theta_sim, lower=0.0, upper=10.0)

# Standardize Y
std_res_Y = standardize(Y_sim)
Y_mean = std_res_Y['mu']  # (1, m)
Y_std = std_res_Y['std']  # (1, m)
Y_sim_stdzd = std_res_Y['standardized']  # (n, m)

# Standardize the transformed parameters
std_res = standardize(theta_sim_unconstrained)
parameters_mean = std_res['mu']   # (1, 5)
parameters_std = std_res['std']     # (1, 5)
parameters_stdzd = std_res['standardized']  # (n, 5)

# ---------------------------
# Generate Observed Data
# ---------------------------
# Sample from the prior to generate parameters for the observed data
obs_params = sample_prior(1)  # shape: (1, 5)
true_params = obs_params.squeeze(0).numpy()  # Save true parameters for plotting

print(f"True parameters: θ₁={true_params[0]:.3f}, θ₂={true_params[1]:.3f}, θ₃={true_params[2]:.3f}, θ₄={true_params[3]:.3f}, σ²={true_params[4]:.3f}")

# Simulate observed data using the sampled parameters
Y_obs = simulator(obs_params, m).squeeze(0)  # shape: (m,)

# Standardize the observed data
Y_obs_stdzd = (Y_obs - Y_mean) / torch.clamp(Y_std, min=1e-6)
if Y_obs_stdzd.dim() == 1:
    Y_obs_stdzd = Y_obs_stdzd.unsqueeze(0)

# Save true parameters
np.savetxt(
    os.path.join(OUTDIR, f"true_params_seed{SEED}.csv"),
    true_params.reshape(1, -1),
    delimiter=",",
    header="theta1,theta2,theta3,theta4,sigma2",
    comments=""
)


# In[75]:


# ---------------------------
# Split Training and Validation Data
# ---------------------------
indices = np.random.permutation(n)
n_train = int(n * 0.9)
train_idx = indices[:n_train]
val_idx = indices[n_train:]
Y_train = Y_sim_stdzd[train_idx]          # (n_train, m)
params_train = parameters_stdzd[train_idx]  # (n_train, 5)
Y_val = Y_sim_stdzd[val_idx]              # (n_val, m)
params_val = parameters_stdzd[val_idx]  # (n_val, 5)

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
# Here, 'features' is the target dimension (5) and 'context_features' is the input dimension (m)
num_components = 6
mdn = MultivariateGaussianMDN(
    features=5,  # Updated to 5 parameters
    context_features=m,
    hidden_net=hidden_net,
    num_components=num_components,
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
clip_max_norm = 6.0  # maximum gradient norm

for epoch in range(1, num_epochs + 1):
    mdn.train()
    total_loss = 0.0
    permutation = np.random.permutation(n_train)
    for i in range(0, n_train, batch_size):
        optimizer.zero_grad()
        batch_idx = permutation[i:i+batch_size]
        Y_batch = Y_train[batch_idx]           # context: (batch, m)
        target_batch = params_train[batch_idx]   # target: (batch, 5)
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
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = mdn.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1

    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")

    if patience_counter >= patience_threshold:
        print(f"Early stopping at epoch {epoch}.")
        break


if best_state is not None:
    mdn.load_state_dict(best_state)
    print("Loaded best MDN weights.")


# In[76]:


##########################
#Sample from the estimated posterior
##########################
mdn.eval()
with torch.no_grad():
    # Sample from MDN (in standardized unconstrained space)
    samples = mdn.sample(n, Y_obs_stdzd)  # shape: (batch, n, 5)
    samples = samples.squeeze(0)  # shape: (n, 5)
    
    # Unstandardize: original = std * sample + mean
    samples_unconstrained = samples * parameters_std + parameters_mean
    
    # Transform back to constrained space [0, 10] for θ and (0, ∞) for σ²
    samples_unstd = transform_to_constrained(samples_unconstrained, lower=0.0, upper=10.0)


# In[77]:


#save mean_list and cov_list for round 1
logits, means, precisions, _, _  = mdn.get_mixture_components(Y_obs_stdzd)
covs = torch.linalg.inv(precisions)
D = torch.diag(parameters_std.squeeze())
D_expanded = D.unsqueeze(0).unsqueeze(0)     # now shape (1, 1, 5, 5) for broadcasting
mean_unstd = parameters_mean + parameters_std * means # shape: (1, num_components, 5)
cov_unstd = D_expanded @ covs @ D_expanded  # shape: (1, num_components, 5, 5)

# Convert to numpy
means_np = mean_unstd[0].detach().cpu().numpy()  # (num_components, 5)
covs_np = cov_unstd[0].detach().cpu().numpy()    # (num_components, 5, 5)
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


# In[79]:


##########################
# Get Ground Truth Posterior via Stan MCMC
##########################

# Prepare data for Stan
x_np = np.linspace(0, 10, m)  # design points
Y_obs_np = Y_obs.numpy()       # observed Y values

stan_data = {
    'N': int(m),              # number of observations
    'x': x_np.tolist(),       # design points (X values)
    'y': Y_obs_np.tolist()    # observed data (Y values)
}

# ---- compile ----
model = CmdStanModel(
    stan_file=str(SCRIPT_DIR / "exp2term.stan")
)



# ---- sample ----
fit = model.sample(
    data=stan_data,
    chains=1,
    iter_warmup=2000,
    iter_sampling=n,
    seed=SEED,
    adapt_delta=0.9,          # helps if there are correlations
    max_treedepth=12
)


# In[83]:


##########################
# MCMC Trace Plots
##########################
# Get the draws for each parameter separately for each chain
mcmc_samples = fit.stan_variables()

# Create 5 separate trace plots
param_names = ['theta1', 'theta2', 'theta3', 'theta4', 'sigma2']

for param_name in param_names:
    plt.figure(figsize=(12, 4))
    
    # Get samples for this parameter
    param_samples = mcmc_samples[param_name]
    
    # Plot the trace
    plt.plot(param_samples[0:5000], color='black', alpha=0.7, linewidth=0.8)
    
    plt.xlabel('Iteration')
    plt.ylabel(param_name)
    plt.title(f'Trace Plot for {param_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    trace_filename = os.path.join(OUTDIR, f"trace_plot_{param_name}_seed{SEED}.png")
    plt.savefig(trace_filename, dpi=150)
    print(f"Saved trace plot for {param_name}")
    
    # Display plot in notebook
    plt.show()

print("All trace plots saved and displayed.")


# In[ ]:


# Extract MCMC samples 
mcmc_samples = fit.stan_variables()

stan_posterior_samples = np.column_stack([
    mcmc_samples['theta1'],
    mcmc_samples['theta2'],
    mcmc_samples['theta3'],
    mcmc_samples['theta4'],
    mcmc_samples['sigma2']
])

print(f"Stan MCMC posterior samples shape: {stan_posterior_samples.shape}")
# Save MCMC samples for reference
np.savetxt(
    os.path.join(OUTDIR, f"posterior_mcmc_seed{SEED}.csv"),
    stan_posterior_samples,
    delimiter=",",
    header="theta1,theta2,theta3,theta4,sigma2",
    comments=""
)


# In[85]:


##########################
# C2ST Test - Round 1 (NPE vs MCMC Ground Truth)
##########################
# Convert NPE samples to numpy arrays
est_samples_np = samples_unstd.detach().cpu().numpy()

# Keep unconstrained version for plotting
est_samples_unconstrained_np = samples_unconstrained.detach().cpu().numpy()

# Transform MCMC samples to unconstrained space for plotting
stan_posterior_samples_tensor = torch.from_numpy(stan_posterior_samples).float()
stan_unconstrained = transform_to_unconstrained(stan_posterior_samples_tensor, lower=0.0, upper=10.0)
stan_unconstrained_np = stan_unconstrained.numpy()

# Transform true parameters to unconstrained space
true_params_tensor = torch.from_numpy(true_params).float().unsqueeze(0)
true_params_unconstrained = transform_to_unconstrained(true_params_tensor, lower=0.0, upper=10.0)
true_params_unconstrained_np = true_params_unconstrained.squeeze(0).numpy()

param_names = ['theta1', 'theta2', 'theta3', 'theta4', 'sigma2']

# Plot marginal distributions in UNCONSTRAINED space
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, param in enumerate(param_names):
    sns.kdeplot(stan_unconstrained_np[:, i], ax=axes[i], label='MCMC', color='blue')
    sns.kdeplot(est_samples_unconstrained_np[:, i], ax=axes[i], label='NPE', color='red')
    # Add vertical line for true parameter value
    axes[i].axvline(true_params_unconstrained_np[i], color='red', linestyle='--', linewidth=2, label='True')
    axes[i].set_title(f"Marginal for {param} (Unconstrained)")
    if i < 4:
        axes[i].set_xlabel(f"{param} (logit-transformed)")
    else:
        axes[i].set_xlabel(f"{param} (log-transformed)")
    axes[i].legend()

# Hide the extra subplot
axes[5].axis('off')

plt.tight_layout()

plot_filename = os.path.join(OUTDIR, f"comparison_plot_round_1_seed{SEED}.png")
plt.savefig(plot_filename)
print(f"Saved comparison plot as {plot_filename}")

# Save NPE posterior samples
np.savetxt(
    os.path.join(OUTDIR, f"posterior_estimated1_seed{SEED}.csv"),
    est_samples_np,
    delimiter=",",
    header="theta1,theta2,theta3,theta4,sigma2",
    comments=""
)


# In[ ]:


# Classifier 2-Sample Test (C2ST) - MCMC vs NPE - uses CONSTRAINED space
acc_per_round = []

# Stack samples and labels (0 = MCMC ground truth, 1 = NPE)
X = np.vstack([stan_posterior_samples, est_samples_np])
y = np.concatenate([np.zeros(len(stan_posterior_samples)), np.ones(len(est_samples_np))])

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
print(f"C2ST accuracy (first round): {acc:.4f}")
print(f"Note: Accuracy close to 0.5 indicates NPE matches MCMC ground truth well.")

np.savetxt(
    os.path.join(OUTDIR, f"acc_per_round_seed{SEED}.csv"),
    acc_per_round,
    delimiter=",",
    header="Accuracy",
    comments=""
)


# In[ ]:


# ---------------------------
# Accumulate first-round data
# ---------------------------
all_theta = [theta_sim_unconstrained]
all_Y = [Y_sim]


# In[ ]:


for rnd in range(2, 31):
    optimizer = torch.optim.Adam(mdn.parameters(), lr=5e-4)  #Re-init optimizer
    print(f"\n=== Round {rnd} ===")

    with torch.no_grad():
        n_proposal = n
        additional_parameters = sample_prior(n_proposal)  # shape: (n, 5)
        
        # Transform to unconstrained space
        additional_parameters_unconstrained = transform_to_unconstrained(additional_parameters, lower=0.0, upper=10.0)
        
        Y_additional = simulator(additional_parameters, m)
        
        all_theta.append(additional_parameters_unconstrained)
        all_Y.append(Y_additional)

    # Step 8: Prepare training data
    theta_all = torch.cat(all_theta, dim=0)
    Y_all = torch.cat(all_Y, dim=0)

    theta_all_std = (theta_all - parameters_mean) / parameters_std
    Y_all_std = (Y_all - Y_mean) / Y_std


    # Split train/validation
    n_total = theta_all_std.shape[0]
    idx_shuffle = torch.randperm(n_total)
    split_idx = int(0.9 * n_total)
    
    theta_train = theta_all_std[idx_shuffle[:split_idx]]
    theta_val = theta_all_std[idx_shuffle[split_idx:]]
    Y_train = Y_all_std[idx_shuffle[:split_idx]]
    Y_val = Y_all_std[idx_shuffle[split_idx:]]

    # Train MDN WITHOUT importance weighting (standard NLL loss)
    best_state = None
    patience_counter = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        mdn.train()
        perm = torch.randperm(len(theta_train))
        epoch_loss = 0.0
        
        for i in range(0, len(theta_train), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch_theta = theta_train[batch_idx]
            batch_Y = Y_train[batch_idx]

            optimizer.zero_grad()
            
            # Standard negative log-likelihood (NO importance weights for TSNPE!)
            log_prob = mdn.log_prob(batch_theta, batch_Y)
            loss = -log_prob.mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdn.parameters(), max_norm=clip_max_norm)
            optimizer.step()
            epoch_loss += loss.item()

        mdn.eval()
        with torch.no_grad():
            val_log_prob = mdn.log_prob(theta_val, Y_val)
            val_loss = -val_log_prob.mean().item()

        if (epoch + 1) % 5 == 0:
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


    # Step 10: Sample from posterior
    posterior_samples_std = mdn.sample(n, Y_obs_stdzd).squeeze(0)
    posterior_samples_unconstrained = posterior_samples_std * parameters_std + parameters_mean
    
    # Transform back to [0, 10] for θ and (0, ∞) for σ² for saving/C2ST
    posterior_samples = transform_to_constrained(posterior_samples_unconstrained, lower=0.0, upper=10.0)
    posterior_np = posterior_samples.detach().cpu().numpy()
    
    # Keep unconstrained for plotting
    posterior_unconstrained_np = posterior_samples_unconstrained.detach().cpu().numpy()



    # Plot marginals in UNCONSTRAINED space with true values
    param_names = ['theta1', 'theta2', 'theta3', 'theta4', 'sigma2']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, param in enumerate(param_names):
        sns.kdeplot(stan_unconstrained_np[:, i], ax=axes[i], label='MCMC', color='blue')
        sns.kdeplot(posterior_unconstrained_np[:, i], ax=axes[i], label='NPE', color='red')
        # Add vertical line for true parameter value
        axes[i].axvline(true_params_unconstrained_np[i], color='red', linestyle='--', linewidth=2, label='True')
        axes[i].set_title(f"Marginal for {param} (Unconstrained)")
        if i < 4:
            axes[i].set_xlabel(f"{param} (logit-transformed)")
        else:
            axes[i].set_xlabel(f"{param} (log-transformed)")
        axes[i].legend()
    
    # Hide the extra subplot
    axes[5].axis('off')

    plt.tight_layout()
    plot_filename = os.path.join(OUTDIR, f"comparison_plot_round_{rnd}_seed{SEED}.png")
    plt.savefig(plot_filename)
    print(f"Saved comparison plot as {plot_filename}")

    # C2ST - uses CONSTRAINED space
    X = np.vstack([stan_posterior_samples, posterior_np])
    y = np.concatenate([np.zeros(len(stan_posterior_samples)), np.ones(len(posterior_np))])
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    acc_per_round.append(acc)
    print(f"C2ST accuracy (round {rnd}): {acc:.4f}")

    #save acc_per_round
    np.savetxt(
        os.path.join(OUTDIR, f"acc_per_round_seed{SEED}.csv"),
        acc_per_round,
        delimiter=",",
        header="Accuracy",
        comments=""  # Prevents NumPy from adding a '#' before the header
    )

    #save the posterior samples
    filename = os.path.join(OUTDIR, f"posterior_estimated{rnd}_seed{SEED}.csv")
    np.savetxt(
        filename,
        posterior_np,
        delimiter=",",
        header="theta1,theta2,theta3,theta4,sigma2",
        comments=""
    )

    #save parameters of mdn fitted
    logits, means, precisions, _, _  = mdn.get_mixture_components(Y_obs_stdzd)
    covs = torch.linalg.inv(precisions)
    D = torch.diag(parameters_std.squeeze())
    D_expanded = D.unsqueeze(0).unsqueeze(0)     # now shape (1, 1, 5, 5) for broadcasting
    mean_unstd = parameters_mean + parameters_std * means # shape: (1, num_components, 5)
    cov_unstd = D_expanded @ covs @ D_expanded  # shape: (1, num_components, 5, 5)

    # Convert to numpy
    means_np = mean_unstd[0].detach().cpu().numpy()  # (num_components, 5)
    covs_np = cov_unstd[0].detach().cpu().numpy()    # (num_components, 5, 5)
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

