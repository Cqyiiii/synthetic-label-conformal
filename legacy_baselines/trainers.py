import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from losses import *
from utils import to_numpy

def train_mean(model, x_tr, y_tr, epochs=100):
    opt = optim.Adam(model.parameters(), lr=2e-3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=1024, shuffle=True)
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            loss = nn.HuberLoss(delta=1.0)(model(bx), by)
            loss.backward()
            opt.step()
    return model

def train_cqr_nd(model, x_tr, y_tr, taus, mode='pinball', epochs=100):
    opt = optim.Adam(model.parameters(), lr=2e-3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=1024, shuffle=True)
    D = y_tr.shape[1]
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            preds = model(bx)
            if mode == 'ald': 
                loss = ald_loss_cqr_nd(preds, by, taus)
            else:
                q_lo = preds[:, :D]; q_hi = preds[:, D:]
                loss = (pinball_loss(q_lo, by, taus[0]) + pinball_loss(q_hi, by, taus[1])).mean()
            loss.backward()
            opt.step()
    return model

def train_cqr_syn(model, x_tr, y_tr_real, y_tr_syn, x_syn, y_syn_large, taus, epochs=100, power_tuner=1):
    """
    Semi-Supervised CQR Trainer.
    Standard CQR trains on the Training Set (not calibration).
    
    x_tr, y_tr_real: Real Training Data
    y_tr_syn:        Synthetic labels for x_tr (Small Syn)
    x_syn, y_syn_large: Synthetic labels for large unlabeled set (Large Syn)
    """
    opt = optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    
    D = y_tr_real.shape[1]
    
    for _ in range(epochs):
        opt.zero_grad()
        
        # --- Helper for CQR Loss (Sum of Pinball for Low/High) ---
        def cqr_loss(x_in, y_target):
            preds = model(x_in)
            q_lo, q_hi = preds[:, :D], preds[:, D:]
            loss_lo = pinball_loss(q_lo, y_target, taus[0])
            loss_hi = pinball_loss(q_hi, y_target, taus[1])
            return (loss_lo + loss_hi).mean()

        # 1. Real Loss (on Training Set)
        loss_real = cqr_loss(x_tr, y_tr_real)
        
        # 2. Synthetic Loss Small (on Training Set)
        loss_syn_small = cqr_loss(x_tr, y_tr_syn)
        
        # 3. Synthetic Loss Large (on Syn Set)
        loss_syn_large = cqr_loss(x_syn, y_syn_large)
        
        # 4. Combine
        loss = loss_real + power_tuner * (loss_syn_large - loss_syn_small)
        
        loss.backward()
        opt.step()
        
    return model

 

def train_rcp_score(model, x_tr, s_tr, tau, mode='pinball', epochs=100):
    opt = optim.Adam(model.parameters(), lr=2e-3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, s_tr), batch_size=1024, shuffle=True)
    model.train()
    for _ in range(epochs):
        for bx, bs in loader:
            opt.zero_grad()
            preds = model(bx)
            if mode == 'ald': loss = ald_loss_rcp(preds, bs, tau)
            elif mode == 'pinball': loss = pinball_loss(preds, bs, tau).mean()
            loss.backward()
            opt.step()
    return model



def train_rcp_score_syn(model, x_n, s_n_real, s_n_syn, x_N, s_N_syn, tau=0.9, power_tuner=1, epochs=100):
    opt = optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        q_preds_n, q_preds_N = model(x_n), model(x_N)
        # Loss combination: Pinball real + (Pinball Syn N - Pinball Syn n)
        loss = pinball_loss(q_preds_n, s_n_real, tau).mean() + power_tuner * (
            pinball_loss(q_preds_N, s_N_syn, tau).mean() - pinball_loss(q_preds_n, s_n_syn, tau).mean()
        )
        loss.backward()
        opt.step()
    return model

def train_plcp_model(model, x_tr, s_tr, target_tau, loss_type='pinball', epochs=100):
    opt = optim.Adam(model.parameters(), lr=2e-3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, s_tr), batch_size=1024, shuffle=True)
    model.train()
    for _ in range(epochs):
        for bx, bs in loader:
            opt.zero_grad()
            h_probs, q_params = model(bx)
            s_exp = bs.repeat(1, q_params.shape[1])
            q_exp = q_params.repeat(bx.shape[0], 1)
            diff = s_exp - q_exp
            elem_loss = torch.max(target_tau * diff, (target_tau - 1) * diff)
            loss = torch.mean(torch.sum(h_probs * elem_loss, dim=1))
            loss.backward()
            opt.step()
    return model


def train_plcp_score_syn(model, x_n, s_n_real, s_n_syn, x_N, s_N_syn, tau=0.9, power_tuner=1, epochs=100):
    """
    Semi-Supervised Trainer for PLCP using synthetic data difference.
    x_n, s_n_real: Real calibration data (small)
    s_n_syn:       Synthetic scores on x_n
    x_N, s_N_syn:  Synthetic scores on large unlabeled data
    """
    opt = optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    
    for _ in range(epochs):
        opt.zero_grad()
        
        # 1. Real Data Loss
        h_n, q_n = model(x_n)
        loss_real = plcp_loss_function(h_n, q_n, s_n_real, tau)
        
        # 2. Synthetic Loss on Real X (Small Batch)
        # Note: model(x_n) returns same h_n, q_n (q is parameter)
        loss_syn_n = plcp_loss_function(h_n, q_n, s_n_syn, tau)
        
        # 3. Synthetic Loss on Unlabeled X (Large Batch)
        h_N, q_N = model(x_N)
        loss_syn_N = plcp_loss_function(h_N, q_N, s_N_syn, tau)
        
        # Combine: Real + (Syn_Large - Syn_Small)
        loss = loss_real + power_tuner * (loss_syn_N - loss_syn_n)
        
        loss.backward()
        opt.step()
    return model

def train_multivariate_gaussian(model, x_tr, y_tr, epochs=100):
    opt = optim.Adam(model.parameters(), lr=2e-3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=1024, shuffle=True)
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            mu, L = model(bx)
            loss = multivariate_nll_loss(mu, L, by)
            loss.backward()
            opt.step()
    return model


def train_multivariate_gaussian_robust(model, x_tr, y_tr, epochs=100):
    # lower learning rate
    opt = optim.Adam(model.parameters(), lr=2e-4)     
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr), 
        batch_size=1024, 
        shuffle=True
    )    
    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            mu, L = model(bx)            
            loss = multivariate_nll_loss_robust(mu, L, by)            
            # NAN detect
            if torch.isnan(loss) or torch.isinf(loss):
                # print(f"Warning: NaN loss in epoch {epoch}, skipping batch.")
                opt.zero_grad() # Clear gradients
                continue            
            loss.backward()            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)            
            opt.step()
    return model


def train_rcp_multi_head(model, x_tr, s_tr, taus_list, epochs=100):
    opt = optim.Adam(model.parameters(), lr=2e-3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, s_tr), batch_size=1024, shuffle=True)
    model.train()
    for _ in range(epochs):
        for bx, bs in loader:
            opt.zero_grad()
            preds = model(bx)
            loss = 0
            for i, t in enumerate(taus_list):
                loss += pinball_loss(preds[:, i:i+1], bs, t).mean()
            loss.backward()
            opt.step()
    return model

def train_three_head_base(model, x_tr, s_tr, taus_list, epochs=100):
    opt = optim.Adam(model.parameters(), lr=2e-3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, s_tr), batch_size=1024, shuffle=True)
    model.train()
    tau_lo, tau_main, tau_hi = taus_list
    for _ in range(epochs):
        for bx, bs in loader:
            opt.zero_grad()
            preds = model(bx)
            loss_elements = pinball_loss(preds[:, 0:1], bs, tau_lo) + \
                            pinball_loss(preds[:, 1:2], bs, tau_main) + \
                            pinball_loss(preds[:, 2:3], bs, tau_hi)
            loss = loss_elements.mean()
            loss.backward()
            opt.step()
    return model

def finetune_main_head_improved(model, x_tr, s_tr, target_tau, epsilon, epochs=100, 
                                mode='vanilla', clip_max=5.0, mix_ratio=0.5, 
                                save_weights_path=None):
    """
    Implements the core 'Colorful Pinball' optimization: 
    Finetunes the main quantile head using density-weighted pinball loss.
    """
    # Freeze shared and auxiliary heads
    for param in model.shared.parameters(): param.requires_grad = False
    for param in model.head_lo_gap.parameters(): param.requires_grad = False
    for param in model.head_hi_gap.parameters(): param.requires_grad = False
    
    tau_lo = max(0.01, target_tau - epsilon)
    tau_hi = min(0.99, target_tau + epsilon)
    
    # Optional: Debug weight statistics
    if save_weights_path is not None:
        model.eval()
        with torch.no_grad():
            preds_all = model(x_tr)
            q_lo, q_hi = preds_all[:, 0:1], preds_all[:, 2:3]
            q_diff = (q_hi - q_lo) + 1e-6
            raw_weights = (tau_hi - tau_lo) / q_diff
            norm_weights = raw_weights / raw_weights.mean()
            
            df_w = pd.DataFrame({
                'q_diff': to_numpy(q_diff).flatten(),
                'raw_weight': to_numpy(raw_weights).flatten(),
                'norm_weight': to_numpy(norm_weights).flatten()
            })
            os.makedirs(os.path.dirname(save_weights_path), exist_ok=True)
            df_w.to_csv(save_weights_path, index=False)
    
    # Training Loop
    opt = optim.Adam(model.head_main.parameters(), lr=2e-3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, s_tr), batch_size=1024, shuffle=True)
    
    model.train() 
    for _ in range(epochs):
        for bx, bs in loader:
            opt.zero_grad()
            preds = model(bx)
            q_lo, q_main, q_hi = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]
            
            with torch.no_grad():
                q_diff = torch.clamp((q_hi - q_lo), min=1e-4)
                density_weight = (tau_hi - tau_lo) / q_diff
                density_weight = density_weight / density_weight.mean() 
                
                if mode == 'clip':
                    density_weight = torch.clamp(density_weight, max=clip_max)
            
            loss_pinball = pinball_loss(q_main, bs, target_tau)
            loss_weighted = (loss_pinball * density_weight).mean()
            loss_original = loss_pinball.mean()
            
            # Loss Mixing: Alpha * Weighted + (1-Alpha) * Original
            final_loss = mix_ratio * loss_weighted + (1.0 - mix_ratio) * loss_original
            
            final_loss.backward()
            opt.step()
            
    # Unfreeze for future use
    for param in model.parameters(): param.requires_grad = True
    return model



def train_three_head_syn(model, x_n, s_n_real, s_n_syn, x_N, s_N_syn, taus_list, epochs=100, power_tuner=1):
    """
    Semi-supervised training for the 3-head auxiliary network.
    """
    opt = optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    tau_lo, tau_main, tau_hi = taus_list
    
    for _ in range(epochs):
        opt.zero_grad()
        
        # 1. Forward passes
        preds_n_real = model(x_n)      # Real input
        preds_n_syn = model(x_n)       # Real input (for syn loss subtraction)
        preds_N_syn = model(x_N)       # Large synthetic input
        
        # 2. Define Loss Calculator
        def compute_3head_loss(preds, targets):
            # Sum of pinball losses for Lo, Main, Hi heads
            l_lo = pinball_loss(preds[:, 0:1], targets, tau_lo).mean()
            l_main = pinball_loss(preds[:, 1:2], targets, tau_main).mean()
            l_hi = pinball_loss(preds[:, 2:3], targets, tau_hi).mean()
            return l_lo + l_main + l_hi

        # 3. Compute Component Losses
        loss_real = compute_3head_loss(preds_n_real, s_n_real)
        loss_syn_small = compute_3head_loss(preds_n_syn, s_n_syn)
        loss_syn_large = compute_3head_loss(preds_N_syn, s_N_syn)
        
        # 4. Combined Semi-Supervised Loss
        loss = loss_real + power_tuner * (loss_syn_large - loss_syn_small)
        
        loss.backward()
        opt.step()
    return model


def finetune_main_head_syn(model, x_n, s_n_real, s_n_syn, x_N, s_N_syn, 
                           target_tau, epsilon, epochs=100, 
                           mode='vanilla', clip_max=5.0, mix_ratio=0.5, power_tuner=1):
    """
    Semi-supervised fine-tuning of the main head using Density-Weighted Pinball Loss.
    """
    # Freeze shared backbone and auxiliary heads
    for param in model.shared.parameters(): param.requires_grad = False
    for param in model.head_lo_gap.parameters(): param.requires_grad = False
    for param in model.head_hi_gap.parameters(): param.requires_grad = False
    
    tau_lo = max(0.01, target_tau - epsilon)
    tau_hi = min(0.99, target_tau + epsilon)
    
    opt = optim.Adam(model.head_main.parameters(), lr=2e-3)
    model.train()
    
    for _ in range(epochs):
        opt.zero_grad()
        
        # --- Helper to compute Weighted Loss ---
        def get_weighted_loss(x_batch, s_batch):
            preds = model(x_batch)
            q_lo, q_main, q_hi = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]
            
            # Compute Weights (no grad needed for weights usually, but dependent on q_lo/q_hi which are frozen)
            with torch.no_grad():
                q_diff = torch.clamp((q_hi - q_lo), min=1e-4)
                density_weight = (tau_hi - tau_lo) / q_diff
                
                # Normalize weights within this batch
                density_weight = density_weight / density_weight.mean()
                
                if mode == 'clip':
                    density_weight = torch.clamp(density_weight, max=clip_max)
            
            # Weighted Pinball Loss
            loss_pinball = pinball_loss(q_main, s_batch, target_tau)
            loss_weighted = (loss_pinball * density_weight).mean()
            loss_original = loss_pinball.mean()
            
            # Mix
            return mix_ratio * loss_weighted + (1.0 - mix_ratio) * loss_original

        # --- Calculate Semi-Supervised Parts ---
        loss_real = get_weighted_loss(x_n, s_n_real)
        loss_syn_small = get_weighted_loss(x_n, s_n_syn)
        loss_syn_large = get_weighted_loss(x_N, s_N_syn)
        
        # Combine
        loss = loss_real + power_tuner * (loss_syn_large - loss_syn_small)
        
        loss.backward()
        opt.step()
            
    # Unfreeze for future use/eval
    for param in model.parameters(): param.requires_grad = True
    return model