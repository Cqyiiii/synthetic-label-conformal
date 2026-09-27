import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.special import comb, gammaln
from sklearn.neighbors import NearestNeighbors

from models import *
from trainers import *
from metrics import *
from utils import to_tensor, to_numpy, DEVICE
from scipy.special import comb


# Split conformal prediction
def run_split(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, alpha):
    D = Y_tr.shape[1]
    mu_net = Net(X_tr.shape[1], D).to(DEVICE)
    train_mean(mu_net, to_tensor(X_tr), to_tensor(Y_tr))
    mu_net.eval()
    with torch.no_grad():
        mu_cal = to_numpy(mu_net(to_tensor(X_cal)))
        scores = np.max(np.abs(Y_cal - mu_cal), axis=1)
        q = np.quantile(scores, np.ceil((1-alpha)*(len(scores)+1))/len(scores))
        
        mu_te = to_numpy(mu_net(to_tensor(X_te)))
        y_lo, y_hi = mu_te - q, mu_te + q
    return get_metrics_nd(Y_te, y_lo, y_hi, X_te, alpha)

# CQR
def run_cqr(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, alpha, mode='pinball'):
    D = Y_tr.shape[1]
    taus = [alpha/2, 1-alpha/2]
    
    if mode == 'ald': 
        model = ALDNet(X_tr.shape[1], 3*D).to(DEVICE)
    else: 
        model = Net(X_tr.shape[1], 2*D).to(DEVICE)
    
    train_cqr_nd(model, to_tensor(X_tr), to_tensor(Y_tr), taus, mode)
    model.eval()
    
    # Calibration
    with torch.no_grad():
        cal_out = model(to_tensor(X_cal))
        q_lo, q_hi = cal_out[:, :D], cal_out[:, D:2*D]
        
        cal_qs = torch.stack([q_lo, q_hi], dim=2)
        ql = torch.min(cal_qs, dim=2).values
        qh = torch.max(cal_qs, dim=2).values
        Y_cal_ts = to_tensor(Y_cal)
        
        # Conformity Score: Signed distance to interval
        resid_vec = torch.maximum(ql - Y_cal_ts, Y_cal_ts - qh)
        scores = resid_vec.max(dim=1).values.cpu().numpy()
        
        n = len(scores)
        q = np.quantile(scores, np.ceil((1-alpha)*(n+1))/n)
        
    # Inference
    with torch.no_grad():
        te_out = model(to_tensor(X_te))
        q_lo_te, q_hi_te = te_out[:, :D], te_out[:, D:2*D]
        te_qs = torch.stack([q_lo_te, q_hi_te], dim=2)
        ql_te = torch.min(te_qs, dim=2).values
        qh_te = torch.max(te_qs, dim=2).values
        
        y_lo = to_numpy(ql_te - q)
        y_hi = to_numpy(qh_te + q)
        
    return get_metrics_nd(Y_te, y_lo, y_hi, X_te, alpha)



def run_cqr_syn(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, X_syn, Y_syn, Y_tr_syn, alpha=0.1):
    """
    Semi-Supervised CQR Runner.
    X_tr, Y_tr: Real Training Set.
    Y_tr_syn: Synthetic labels for X_tr (required for semi-supervised control term).
    X_syn, Y_syn: Large synthetic dataset.
    """
    D = Y_tr.shape[1]
    taus = [alpha/2, 1-alpha/2]
    
    # 1. Train CQR Net (Semi-Supervised)
    # Output dim is 2*D (Lower and Upper quantiles for each dimension)
    model = Net(X_tr.shape[1], 2*D).to(DEVICE)
    
    train_cqr_syn(
        model, 
        to_tensor(X_tr), to_tensor(Y_tr), to_tensor(Y_tr_syn),
        to_tensor(X_syn), to_tensor(Y_syn),
        taus, epochs=200 # CQR usually needs more epochs
    )
    
    model.eval()
    
    # 2. Calibration (Standard CQR)
    with torch.no_grad():
        cal_out = model(to_tensor(X_cal))
        q_lo, q_hi = cal_out[:, :D], cal_out[:, D:]
        
        # Stack for easier min/max operations across dims if D > 1
        cal_qs = torch.stack([q_lo, q_hi], dim=2)
        ql = torch.min(cal_qs, dim=2).values
        qh = torch.max(cal_qs, dim=2).values
        Y_cal_ts = to_tensor(Y_cal)
        
        # Conformity Score: Signed distance to interval
        # S = max(q_lo - y, y - q_hi)
        resid_vec = torch.maximum(ql - Y_cal_ts, Y_cal_ts - qh)
        scores = resid_vec.max(dim=1).values.cpu().numpy()
        
        n = len(scores)
        q = np.quantile(scores, np.ceil((1-alpha)*(n+1))/n)
        
    # 3. Inference
    with torch.no_grad():
        te_out = model(to_tensor(X_te))
        q_lo_te, q_hi_te = te_out[:, :D], te_out[:, D:]
        
        te_qs = torch.stack([q_lo_te, q_hi_te], dim=2)
        ql_te = torch.min(te_qs, dim=2).values
        qh_te = torch.max(te_qs, dim=2).values
        
        # Apply conformal correction q
        y_lo = to_numpy(ql_te - q)
        y_hi = to_numpy(qh_te + q)
        
    return get_metrics_nd(Y_te, y_lo, y_hi, X_te, alpha)





# RCP 
def run_rcp(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, alpha, mode='pinball'):
    D = Y_tr.shape[1]
    mu_net = Net(X_tr.shape[1], D).to(DEVICE)
    train_mean(mu_net, to_tensor(X_tr), to_tensor(Y_tr))
    mu_net.eval()
    
    with torch.no_grad():
        mu_cal = to_numpy(mu_net(to_tensor(X_cal)))
        S_cal = np.max(np.abs(Y_cal - mu_cal), axis=1)
        mu_te = to_numpy(mu_net(to_tensor(X_te)))
        
    X_est, X_conf, S_est, S_conf = train_test_split(X_cal, S_cal, test_size=0.5, random_state=42)
    r_net = ALDNet(X_tr.shape[1], 2).to(DEVICE) if mode=='ald' else Net(X_tr.shape[1], 1).to(DEVICE)
    train_rcp_score(r_net, to_tensor(X_est), to_tensor(S_est.reshape(-1,1)), 1-alpha, mode)
    r_net.eval()
    
    with torch.no_grad():
        tau_conf = to_numpy(r_net(to_tensor(X_conf)))[:, 0].flatten()
        tau_te = to_numpy(r_net(to_tensor(X_te)))[:, 0].flatten()
        scores = S_conf - tau_conf
        q = np.quantile(scores, np.ceil((1-alpha)*(len(scores)+1))/len(scores))
        width = (tau_te + q)[:, None]
        y_lo, y_hi = mu_te - width, mu_te + width
    return get_metrics_nd(Y_te, y_lo, y_hi, X_te, alpha)




def run_rcp_syn(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, X_ext, Y_ext_syn, Y_cal_syn, alpha=0.1):
    D = Y_tr.shape[1]
    
    # 1. Train Mean Estimator
    mu_net = Net(X_tr.shape[1], D).to(DEVICE)
    train_mean(mu_net, to_tensor(X_tr), to_tensor(Y_tr))
    mu_net.eval()
    
    with torch.no_grad():
        mu_cal = to_numpy(mu_net(to_tensor(X_cal)))
        mu_ext = to_numpy(mu_net(to_tensor(X_ext)))
        mu_te = to_numpy(mu_net(to_tensor(X_te)))
        
        S_cal = np.max(np.abs(Y_cal - mu_cal), axis=1)
        
        # We do synthetic augmentation only on the part for quantile regression 
        # Splitting the passed "cal" set into QR training and Conformal Calibration
        X_qr, X_conf, S_qr, S_conf, Y_qr_syn, _ = train_test_split(X_cal, S_cal, Y_cal_syn, test_size=0.5, random_state=42)                
        
        mu_qr = to_numpy(mu_net(to_tensor(X_qr)))
        S_qr_syn = np.max(np.abs(Y_qr_syn - mu_qr), axis=1)
        S_ext_syn = np.max(np.abs(Y_ext_syn - mu_ext), axis=1)                
    
    # 2. Train Score Network (with synthetic data)
    r_net = Net(X_tr.shape[1], 1).to(DEVICE)
    train_rcp_score_syn(
        r_net, 
        to_tensor(X_qr), to_tensor(S_qr.reshape(-1,1)), to_tensor(S_qr_syn.reshape(-1,1)), 
        to_tensor(X_ext), to_tensor(S_ext_syn.reshape(-1,1)), 
        1-alpha
    )
    r_net.eval()
    
    # 3. Conformalize
    with torch.no_grad():
        tau_conf = to_numpy(r_net(to_tensor(X_conf)))[:, 0].flatten()
        tau_te = to_numpy(r_net(to_tensor(X_te)))[:, 0].flatten()
        
        scores = S_conf - tau_conf
        q = np.quantile(scores, np.ceil((1-alpha)*(len(scores)+1))/len(scores))
        
        width = (tau_te + q)[:, None]
        y_lo, y_hi = mu_te - width, mu_te + width
        
    return get_metrics_nd(Y_te, y_lo, y_hi, X_te, alpha)




# --- Helper for SPI (Numerically Stable) ---
def log_comb(n, k):
    """
    Computes log(nCk) using log-gamma function to avoid overflow.
    log(n!) - log(k!) - log((n-k)!)
    """
    # handle broadcasting if n or k are arrays
    # Ensure inputs are valid for gammaln (inputs > 0 usually, but gammaln(1)=0)
    # Using the property: gamma(n+1) = n!
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

def compute_spi_windows(m, N, beta):
    """
    Computes the window indices R- and R+ for SPI.
    Uses log-space arithmetic to prevent overflow for large N.
    """
    R_minus = np.zeros(m + 1, dtype=int)
    R_plus = np.zeros(m + 1, dtype=int)
    
    # Pre-calculate the denominator log: log(C(N+m+1, m+1))
    log_denom = log_comb(N + m + 1, m + 1)
    
    # ks range: k can technically go from 1 to N+1 based on the combinatorics bounds
    ks = np.arange(1, N + 2)
    
    for r in range(1, m + 2):
        # We need to compute PMF: 
        # P(k) = C(k+r-2, r-1) * C(N+m-k-r+2, m-r+1) / Denom
        
        # 1. Compute inputs for combinations
        n1 = ks + r - 2
        k1 = r - 1
        
        n2 = N + m - ks - r + 2
        k2 = m - r + 1
        
        # 2. Filter invalid combinations where n < k (probability is 0 there)
        # valid indices mask
        valid_mask = (n1 >= k1) & (n2 >= k2)
        
        # Initialize log_pmf with -inf (which becomes 0 after exp)
        log_pmf = np.full_like(ks, -np.inf, dtype=float)
        
        if np.any(valid_mask):
            # 3. Compute Log Probability
            # log(P) = log(C1) + log(C2) - log(Denom)
            term1_log = log_comb(n1[valid_mask], k1)
            term2_log = log_comb(n2[valid_mask], k2)
            
            log_pmf[valid_mask] = term1_log + term2_log - log_denom
            
        # 4. Exponentiate to get actual probabilities
        # Subtract max for stability if needed, but here values should be <= 0 (probabilities)
        pmf = np.exp(log_pmf)
        
        # Normalize to ensure sum is exactly 1.0 (corrects tiny numerical drift)
        pmf_sum = np.sum(pmf)
        if pmf_sum > 0:
            pmf /= pmf_sum
            
        cdf = np.cumsum(pmf)
        
        # 5. Determine R indices
        # R_r^- = max {t : F(t-1) <= beta/2}
        # In our array 'cdf', index 'i' corresponds to k = ks[i].
        # So cdf[i] is F(ks[i]).
        # We want to find index i such that F(ks[i]) <= beta/2 is NOT satisfied by i+1?
        # Actually, let's stick to the definition:
        # Find largest index 'i' where CDF[i] <= beta/2.
        
        valid_minus = np.where(cdf <= beta/2)[0]
        if len(valid_minus) == 0:
            # If even the first element > beta/2
            r_min = 1
        else:
            # ks[valid_minus[-1]] is the k value.
            # The formula usually implies we take the next integer if strictly less?
            # SPI paper: R- = max{k | CDF(k-1) <= beta/2}
            # The values in 'ks' are integers k.
            # Our cdf[i] is Prob(K <= ks[i]).
            # We want largest k such that Prob(K <= k-1) <= beta/2.
            
            # Let's map indices:
            # If valid_minus[-1] is index 'i', it means P(K <= ks[i]) <= beta/2.
            # Since P(K <= k) is monotonic, this certainly implies P(K <= ks[i]-1) <= beta/2.
            # So ks[i] is a valid candidate.
            # What about ks[i]+1? We check P(K <= ks[i]). It is <= beta/2.
            # So k = ks[i]+1 is the R- value (because CDF(k-1) = CDF(ks[i]) <= beta/2).
            r_min = ks[valid_minus[-1]] + 1
            
        # R_r^+ = min {k : F(k) >= 1 - beta/2}
        valid_plus = np.where(cdf >= 1 - beta/2)[0]
        if len(valid_plus) == 0:
            r_max = N + 1
        else:
            r_max = ks[valid_plus[0]]
            
        R_minus[r-1] = r_min
        R_plus[r-1] = r_max
        
    return R_minus, R_plus


def run_spi(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, X_syn, Y_syn, alpha=0.1, beta=0.4):
    """
    Synthetic-powered Predictive Inference (SPI)
    Paper: "Synthetic-Powered Predictive Inference" (NeurIPS 2025)
    
    Uses synthetic data to improve calibration efficiency.
    """
    D = Y_tr.shape[1]
    
    # 1. Train Mean Estimator (Standard)
    mu_net = Net(X_tr.shape[1], D).to(DEVICE)
    train_mean(mu_net, to_tensor(X_tr), to_tensor(Y_tr))
    mu_net.eval()
    
    with torch.no_grad():
        mu_cal = to_numpy(mu_net(to_tensor(X_cal)))
        mu_te  = to_numpy(mu_net(to_tensor(X_te)))
        mu_syn = to_numpy(mu_net(to_tensor(X_syn)))
        
        # 2. Compute Nonconformity Scores (Residuals)
        # Real Calibration Scores
        S_cal = np.max(np.abs(Y_cal - mu_cal), axis=1)
        # Synthetic Scores (using synthetic X and synthetic Y)
        S_syn = np.max(np.abs(Y_syn - mu_syn), axis=1)
        
        # Sort scores
        S_cal_sorted = np.sort(S_cal)
        S_syn_sorted = np.sort(S_syn)
        
        m = len(S_cal)
        N = len(S_syn)
        
        # 3. SPI Core Logic
        # A. Compute Windows (R- and R+)
        R_minus, R_plus = compute_spi_windows(m, N, beta)
        
        # B. Compute Q'_{1-alpha} (Threshold on synthetic data)
        # Corresponds to ceiling((1-alpha)*(N+1)) + 1
        # In 0-indexed python:
        idx_q_prime = int(np.ceil((1 - alpha) * (N + 1))) # The +1 in paper implies next element
        # Clamp to bounds
        idx_q_prime = min(idx_q_prime, N - 1)
        Q_prime_syn = S_syn_sorted[idx_q_prime]
        
        # C. Compute \tilde{R}^- and \tilde{R}^+ (Eq 11 in paper)
        # max {r : R_r^\pm <= ceiling((1-alpha)(N+1))}
        cutoff_idx = int(np.ceil((1 - alpha) * (N + 1)))
        
        # Find max r such that R_minus[r] <= cutoff
        # R arrays are size m+1. Indices 0..m
        
        r_bar_minus = 0
        r_bar_plus = 0
        
        # Check all r from 1 to m+1 (indices 0 to m)
        for i in range(m + 1):
            if R_minus[i] <= cutoff_idx:
                r_bar_minus = i # Store index (which is r-1)
            if R_plus[i] <= cutoff_idx:
                r_bar_plus = i
                
        # D. Compute Final SPI Threshold Q (Eq 10)
        # S_(r) uses 1-based indexing in paper. In python S_cal_sorted[i] is (i+1)-th.
        # We need S_(r_bar_minus) -> S_cal_sorted[r_bar_minus-1] if 1-based?
        # Actually, let's align carefully.
        # If r=1, we want S_(1) -> S_cal_sorted[0]. So index is r-1.
        # Note: r ranges 1..m+1. S_cal has only m elements. S_(m+1) = infinity.
        
        def get_S_r(idx_0):
            if idx_0 < m:
                return S_cal_sorted[idx_0]
            else:
                return np.inf
        
        term1 = min(Q_prime_syn, get_S_r(r_bar_minus))
        term2 = get_S_r(r_bar_plus)
        
        Q_spi = max(term1, term2)
        
        # 4. Inference
        y_lo = mu_te - Q_spi
        y_hi = mu_te + Q_spi
        
    return get_metrics_nd(Y_te, y_lo, y_hi, X_te, alpha)

import numpy as np
from sklearn.neighbors import NearestNeighbors


def run_semicp(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, X_syn, Y_syn, Y_cal_syn, alpha=0.1,):
    """
    SemiCP-style NNM for regression with synthetic-label proxy scores.

    This implements the NNM matching rule using *pseudo-scores*:
        j(i) = argmin_j | S_syn_proxy[i] - S_cal_proxy[j] |,

    and bias-corrects proxy scores by adding the neighbor's (true - proxy) gap:
        S_syn_est[i] = S_syn_proxy[i] + (S_cal_true[j(i)] - S_cal_proxy[j(i)]).

    Notes:
    - This is NNM-style (match on pseudo-score), consistent with SemiCP's Eq.(4) matching spirit,
      but the proxy here comes from synthetic labels rather than model pseudo-labels.
    - No clipping is applied by default (clipping is a modification).
    """
    D = Y_tr.shape[1]

    # 1. Train mean estimator on labeled training set
    mu_net = Net(X_tr.shape[1], D).to(DEVICE)
    train_mean(mu_net, to_tensor(X_tr), to_tensor(Y_tr))
    mu_net.eval()

    with torch.no_grad():
        mu_cal = to_numpy(mu_net(to_tensor(X_cal)))
        mu_te  = to_numpy(mu_net(to_tensor(X_te)))
        mu_syn = to_numpy(mu_net(to_tensor(X_syn)))

        # 2. True calibration scores
        #    S_cal_true[j] = s(x_j, y_j)
        S_cal_true = np.max(np.abs(Y_cal - mu_cal), axis=1)

        # 3. Proxy calibration scores using synthetic labels for the *same* calibration X
        #    S_cal_proxy[j] = s(x_j, y_syn_j)
        S_cal_proxy = np.max(np.abs(Y_cal_syn - mu_cal), axis=1)

        # 4. Proxy scores on synthetic/unlabeled X using their synthetic labels
        #    S_syn_proxy[i] = s(x~_i, y_syn_i)
        S_syn_proxy = np.max(np.abs(Y_syn - mu_syn), axis=1)

        # 5. NNM: nearest neighbor based on pseudo-score (scalar)
        #    Fit NN in 1D pseudo-score space of calibration proxy scores
        cal_proxy_1d = S_cal_proxy.reshape(-1, 1)
        syn_proxy_1d = S_syn_proxy.reshape(-1, 1)

        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(cal_proxy_1d)
        _, indices = nn.kneighbors(syn_proxy_1d)
        matched = indices.ravel()

        # 6. Bias correction: add neighbor's (true - proxy) gap
        #    S_syn_est = S_syn_proxy + (S_cal_true_neighbor - S_cal_proxy_neighbor)
        gap_neighbor = S_cal_true[matched] - S_cal_proxy[matched]
        S_syn_estimated = S_syn_proxy + gap_neighbor

        # Optional modification (NOT in SemiCP): enforce non-negativity
        # S_syn_estimated = np.maximum(0.0, S_syn_estimated)

        # 7. Pool calibration true scores + estimated synthetic scores, then conformal quantile
        S_combined = np.concatenate([S_cal_true, S_syn_estimated], axis=0)

        n = len(S_cal_true)
        N = len(S_syn_estimated)

        q_level = np.ceil((n + N + 1) * (1 - alpha)) / (n + N)
        q_level = min(1.0, max(0.0, q_level))

        Q_semicp = np.quantile(S_combined, q_level, method="higher")

        # 8. Prediction interval
        y_lo = mu_te - Q_semicp
        y_hi = mu_te + Q_semicp

    return get_metrics_nd(Y_te, y_lo, y_hi, X_te, alpha)












































# Partition learning conformal prediction
def run_plcp(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, alpha, n_groups, score_type):
    D = Y_tr.shape[1]
    mu_net = Net(X_tr.shape[1], D).to(DEVICE)
    train_mean(mu_net, to_tensor(X_tr), to_tensor(Y_tr))
    mu_net.eval()
    
    with torch.no_grad():
        mu_cal = to_numpy(mu_net(to_tensor(X_cal)))
        S_cal = np.max(np.abs(Y_cal - mu_cal), axis=1)
        mu_te = to_numpy(mu_net(to_tensor(X_te)))
        
    X_ptr, X_pcal, S_ptr, S_pcal = train_test_split(X_cal, S_cal, test_size=0.5, random_state=42)
    plcp = PLCPNet(X_tr.shape[1], n_groups).to(DEVICE)
    train_plcp_model(plcp, to_tensor(X_ptr), to_tensor(S_ptr.reshape(-1,1)), 1-alpha, score_type)
    plcp.eval()
    
    with torch.no_grad():
        probs, _ = plcp(to_tensor(X_pcal))
        grps = torch.argmax(probs, dim=1).cpu().numpy()
        g_q = np.quantile(S_pcal, np.ceil((1-alpha)*(len(S_pcal)+1))/len(S_pcal))
        
        q_dict = {
            k: np.quantile(S_pcal[grps==k], np.ceil((1-alpha)*(len(S_pcal[grps==k])+1))/len(S_pcal[grps==k])) 
            if np.sum(grps==k) > 10 else g_q 
            for k in range(n_groups)
        }
        
        probs_te, _ = plcp(to_tensor(X_te))
        grps_te = torch.argmax(probs_te, dim=1).cpu().numpy()
        q_te = np.array([q_dict.get(g, g_q) for g in grps_te])[:, None]
        y_lo, y_hi = mu_te - q_te, mu_te + q_te
    return get_metrics_nd(Y_te, y_lo, y_hi, X_te, alpha)



def run_plcp_syn(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, X_ext, Y_ext_syn, Y_cal_syn, alpha=0.1, n_groups=20):
    D = Y_tr.shape[1]
    
    # 1. Train Mean Estimator
    mu_net = Net(X_tr.shape[1], D).to(DEVICE)
    train_mean(mu_net, to_tensor(X_tr), to_tensor(Y_tr))
    mu_net.eval()
    
    with torch.no_grad():
        mu_cal = to_numpy(mu_net(to_tensor(X_cal)))
        mu_ext = to_numpy(mu_net(to_tensor(X_ext))) # Mean pred on large unlabeled set
        mu_te = to_numpy(mu_net(to_tensor(X_te)))
        
        # Real Scores on Calibration
        S_cal = np.max(np.abs(Y_cal - mu_cal), axis=1)
        
        # Split Calibration into Partition Train (ptr) and Partition Calibration (pcal)
        # We perform semi-supervised learning on the 'ptr' part
        X_ptr, X_pcal, S_ptr, S_pcal, Y_ptr_syn, _ = train_test_split(
            X_cal, S_cal, Y_cal_syn, test_size=0.5, random_state=42
        )
        
        # Calculate Synthetic Scores
        # A. On X_ptr (small synthetic counterpart)
        mu_ptr = to_numpy(mu_net(to_tensor(X_ptr)))
        S_ptr_syn = np.max(np.abs(Y_ptr_syn - mu_ptr), axis=1)
        
        # B. On X_ext (large synthetic set)
        S_ext_syn = np.max(np.abs(Y_ext_syn - mu_ext), axis=1)

    # 2. Train PLCP-Syn Model
    # We co-train the partition function h(x) and quantiles q
    plcp = PLCPNet(X_tr.shape[1], n_groups).to(DEVICE)
    
    train_plcp_score_syn(
        plcp, 
        to_tensor(X_ptr), to_tensor(S_ptr.reshape(-1,1)), 
        to_tensor(S_ptr_syn.reshape(-1,1)),
        to_tensor(X_ext), to_tensor(S_ext_syn.reshape(-1,1)),
        tau=1-alpha, power_tuner=1.0, epochs=100
    )

    plcp.eval()
    
    # 3. Conformalize (Standard PLCP procedure on the hold-out pcal set)
    with torch.no_grad():
        # Assign groups for calibration data
        probs, _ = plcp(to_tensor(X_pcal))
        grps = torch.argmax(probs, dim=1).cpu().numpy()
        
        # Global quantile fallback
        g_q = np.quantile(S_pcal, np.ceil((1-alpha)*(len(S_pcal)+1))/len(S_pcal))
        
        # Group-wise quantiles
        q_dict = {
            k: np.quantile(S_pcal[grps==k], np.ceil((1-alpha)*(len(S_pcal[grps==k])+1))/len(S_pcal[grps==k])) 
            if np.sum(grps==k) > 10 else g_q 
            for k in range(n_groups)
        }
        
        # Inference on Test
        probs_te, _ = plcp(to_tensor(X_te))
        grps_te = torch.argmax(probs_te, dim=1).cpu().numpy()
        q_te = np.array([q_dict.get(g, g_q) for g in grps_te])[:, None]
        y_lo, y_hi = mu_te - q_te, mu_te + q_te
        
    return get_metrics_nd(Y_te, y_lo, y_hi, X_te, alpha)



# CPCP with better stability, loss mixing + weight clip
def run_rcp_density_improved(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, alpha, 
                             epsilon=0.02, mode='vanilla', clip_max=5.0, mix_ratio=1.0,
                             dataset_name="unknown", seed=0):
    """
    Main Runner for Colorful Pinball Conformal Prediction (CPCP).
    """
    D = Y_tr.shape[1]
    mu_net = Net(X_tr.shape[1], D).to(DEVICE)
    train_mean(mu_net, to_tensor(X_tr), to_tensor(Y_tr))
    mu_net.eval()
    
    with torch.no_grad():
        mu_cal = to_numpy(mu_net(to_tensor(X_cal)))
        S_cal = np.max(np.abs(Y_cal - mu_cal), axis=1)
        mu_te = to_numpy(mu_net(to_tensor(X_te)))
    
    # Split calibration data for three stages: 1. Aux Quantiles, 2. Weight Est & Finetune, 3. Conformalization
    n = len(X_cal)
    idx1 = int(0.4 * n)
    idx2 = int(0.8 * n)
    perm = np.random.permutation(n)
    X_cal, S_cal = X_cal[perm], S_cal[perm]
    
    X_est1, S_est1 = X_cal[:idx1], S_cal[:idx1]
    X_est2, S_est2 = X_cal[idx1:idx2], S_cal[idx1:idx2]
    X_score, S_score = X_cal[idx2:], S_cal[idx2:]
    
    target_q = 1 - alpha
    taus_list = [max(0.01, target_q - epsilon), target_q, min(0.99, target_q + epsilon)]
    
    # 1. Train auxiliary quantiles
    r_net = MonotonicThreeHeadNet(X_tr.shape[1]).to(DEVICE)
    train_three_head_base(r_net, to_tensor(X_est1), to_tensor(S_est1.reshape(-1,1)), taus_list, epochs=200)
    
    # 2. Fine-tune main head with density weights (CPCP core)
    finetune_main_head_improved(r_net, to_tensor(X_est2), to_tensor(S_est2.reshape(-1,1)), 
                                target_tau=target_q, epsilon=epsilon, epochs=200,
                                mode=mode, clip_max=clip_max, mix_ratio=mix_ratio,
                                save_weights_path=None)
    
    r_net.eval()
    with torch.no_grad():
        tau_conf = to_numpy(r_net(to_tensor(X_score)))[:, 1].flatten()
        tau_te = to_numpy(r_net(to_tensor(X_te)))[:, 1].flatten()
        tau_conf, tau_te = np.maximum(tau_conf, 1e-4), np.maximum(tau_te, 1e-4)
        
        scores = S_score - tau_conf
        q = np.quantile(scores, np.ceil((1-alpha)*(len(scores)+1))/len(scores))
        
        width = (tau_te + q)[:, None]
        y_lo, y_hi = mu_te - width, mu_te + width
        
    return get_metrics_nd(Y_te, y_lo, y_hi, X_te, alpha)



def run_cpcp_syn(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, X_ext, Y_ext_syn, Y_cal_syn, alpha=0.1, 
                 epsilon=0.02, mode='vanilla', clip_max=5.0, mix_ratio=0.5):
    """
    Semi-Supervised CPCP Runner.
    X_ext, Y_ext_syn: Large synthetic dataset (N).
    Y_cal_syn: Synthetic labels corresponding to X_cal (n).
    """
    D = Y_tr.shape[1]
    
    # 1. Train Mean Estimator (Supervised only)
    mu_net = Net(X_tr.shape[1], D).to(DEVICE)
    train_mean(mu_net, to_tensor(X_tr), to_tensor(Y_tr))
    mu_net.eval()
    
    with torch.no_grad():
        mu_cal = to_numpy(mu_net(to_tensor(X_cal)))
        mu_te = to_numpy(mu_net(to_tensor(X_te)))
        mu_ext = to_numpy(mu_net(to_tensor(X_ext)))
        
        # Real Scores
        S_cal = np.max(np.abs(Y_cal - mu_cal), axis=1)
        
        # Synthetic Scores (on Large Ext set)
        S_ext_syn = np.max(np.abs(Y_ext_syn - mu_ext), axis=1)
        
        # Synthetic Scores (on Small Cal set)
        # Note: Y_cal_syn comes from RF predicting on X_cal
        S_cal_syn = np.max(np.abs(Y_cal_syn - mu_cal), axis=1)

    # 2. Split Calibration Data for CPCP (40% / 40% / 20%)
    # We must split X_cal, S_cal AND S_cal_syn consistently
    n = len(X_cal)
    idx1 = int(0.4 * n)
    idx2 = int(0.8 * n)
    
    perm = np.random.permutation(n)
    
    # Slice Real Data
    X_cal_p, S_cal_p = X_cal[perm], S_cal[perm]
    S_cal_syn_p = S_cal_syn[perm] # Slice Syn data same way
    
    # Part 1: Aux Training (40%)
    X_est1 = X_cal_p[:idx1]
    S_est1_real = S_cal_p[:idx1]
    S_est1_syn = S_cal_syn_p[:idx1]
    
    # Part 2: Weight Est & Fine-tuning (40%)
    X_est2 = X_cal_p[idx1:idx2]
    S_est2_real = S_cal_p[idx1:idx2]
    S_est2_syn = S_cal_syn_p[idx1:idx2]
    
    # Part 3: Conformalization (20%) - STRICTLY REAL DATA ONLY
    X_score = X_cal_p[idx2:]
    S_score = S_cal_p[idx2:]
    
    target_q = 1 - alpha
    taus_list = [max(0.01, target_q - epsilon), target_q, min(0.99, target_q + epsilon)]

    # 3. Train Auxiliary Quantiles (Semi-Supervised)
    # Uses Part 1 (Real + Syn Small) vs Ext (Syn Large)
    r_net = MonotonicThreeHeadNet(X_tr.shape[1]).to(DEVICE)
    
    train_three_head_syn(
        r_net,
        to_tensor(X_est1), to_tensor(S_est1_real.reshape(-1,1)), to_tensor(S_est1_syn.reshape(-1,1)),
        to_tensor(X_ext), to_tensor(S_ext_syn.reshape(-1,1)),
        taus_list, epochs=100
    )
    
    # 4. Fine-tune Main Head (Semi-Supervised)
    # Uses Part 2 (Real + Syn Small) vs Ext (Syn Large)
    finetune_main_head_syn(
        r_net, 
        to_tensor(X_est2), to_tensor(S_est2_real.reshape(-1,1)), to_tensor(S_est2_syn.reshape(-1,1)),
        to_tensor(X_ext), to_tensor(S_ext_syn.reshape(-1,1)),
        target_tau=target_q, epsilon=epsilon, epochs=100,
        mode=mode, clip_max=clip_max, mix_ratio=mix_ratio
    )
    
    # 5. Conformalize (Standard) on Part 3
    r_net.eval()
    with torch.no_grad():
        tau_conf = to_numpy(r_net(to_tensor(X_score)))[:, 1].flatten()
        tau_te = to_numpy(r_net(to_tensor(X_te)))[:, 1].flatten()
        
        # Stability: Ensure positive estimates
        tau_conf, tau_te = np.maximum(tau_conf, 1e-4), np.maximum(tau_te, 1e-4)
        
        scores = S_score - tau_conf
        q = np.quantile(scores, np.ceil((1-alpha)*(len(scores)+1))/len(scores))
        
        width = (tau_te + q)[:, None]
        y_lo, y_hi = mu_te - width, mu_te + width
        
    return get_metrics_nd(Y_te, y_lo, y_hi, X_te, alpha)