import numpy as np

def chromatic_removal(I_in, L_hor=7, L_ver=4, rho=(-0.25, 1.375, -0.125), tau=15, gamma_1=128, gamma_2=64, use_yuv=False):
    ## Introduction
    R_in = I_in[..., 0]
    G_in = I_in[..., 1]
    B_in = I_in[..., 2]
    Y_in = 0.299 * R_in + 0.587 * G_in + 0.114 * B_in

    ## Filtering
    # Horizontal pass
    K_r_hor, K_rTI_hor, R_max_hor, R_min_hor = ti_and_ca_filtering1D(R_in, G_in, Y_in, L_hor, rho=rho, tau=tau, alpha_X=0.5)
    K_b_hor, K_bTI_hor, B_max_hor, B_min_hor = ti_and_ca_filtering1D(B_in, G_in, Y_in, L_hor, rho=rho, tau=tau, alpha_X=1.0)

    # Vertical pass
    K_r_ver, K_rTI_ver, R_max_ver, R_min_ver = ti_and_ca_filtering1D(R_in.T, G_in.T, Y_in.T, L_ver, rho=rho, tau=tau, alpha_X=0.5)
    K_b_ver, K_bTI_ver, B_max_ver, B_min_ver = ti_and_ca_filtering1D(B_in.T, G_in.T, Y_in.T, L_ver, rho=rho, tau=tau, alpha_X=1.0)
    K_r_ver = K_r_ver.T
    K_b_ver = K_b_ver.T
    K_rTI_ver = K_rTI_ver.T
    K_bTI_ver = K_bTI_ver.T
    R_max_ver = R_max_ver.T
    B_max_ver = B_max_ver.T
    R_min_ver = R_min_ver.T
    B_min_ver = B_min_ver.T

    ## Arbitration
    start = time.time()
    # Build the 2D images from the vertically and the horizontally FC filtered images (Eqs. (16))
    # Kb
    K_b = np.array(K_b_ver)
    mask = np.abs(K_b_hor) < np.abs(K_b_ver)
    K_b[mask] = K_b_hor[mask]
    #Kr
    K_r = np.array(K_r_ver)
    mask = np.abs(K_r_hor) < np.abs(K_r_ver)
    K_r[mask] = K_r_hor[mask]

    # Build the 2D images from the vertically and the horizontally TI filtered images (Eqs. (18))
    # Kb
    K_bTI = np.array(K_bTI_ver)
    B_max = np.array(B_max_ver)
    B_min = np.array(B_min_ver)
    mask = np.abs(K_bTI_hor) < np.abs(K_bTI_ver)
    K_bTI[mask] = K_bTI_hor[mask]
    B_max[mask] = B_max_hor[mask]
    B_min[mask] = B_min_hor[mask]

    # Kr
    K_rTI = np.array(K_rTI_ver)
    R_max = np.array(R_max_ver)
    R_min = np.array(R_min_ver)
    mask = np.abs(K_rTI_hor) < np.abs(K_rTI_ver)
    K_rTI[mask] = K_rTI_hor[mask]
    R_max[mask] = R_max_hor[mask]
    R_min[mask] = R_min_hor[mask]

    # Contrast arbitration
    K_rout = arbitration(K_r, K_rTI, R_in, G_in, R_max, R_min, beta_X=1.0, L_hor=L_hor, L_ver=L_ver, gamma_1=gamma_1, gamma_2=gamma_2)
    K_bout = arbitration(K_b, K_bTI, B_in, G_in, B_max, B_min, beta_X=0.25, L_hor=L_hor, L_ver=L_ver, gamma_1=gamma_1, gamma_2=gamma_2)

    K_rout = K_rTI
    K_bout = K_bTI

    # Final RGB conversion (Eq. (24))
    if use_yuv:
        I_out = np.stack([Y_in + 1.13983 * K_rout,
                          Y_in - 0.39465 * K_bout - 0.58060 * K_rout,
                          Y_in + 2.03211 * K_bout], axis=-1)
    else:
        I_out = np.stack([K_rout + G_in, G_in, K_bout + G_in], axis=-1)

    return I_out


def ti_and_ca_filtering1D(X_in, G_in, Y_in, L, rho, alpha_X, tau):
    LL = 2 * L + 1
    X_in_padded = np.pad(X_in, [(0, 0), (L, L)], mode='edge')  # (M, N+2*L)
    G_in_padded = np.pad(G_in, [(0, 0), (L, L)], mode='edge')  # (M, N+2*L)
    Y_in_padded = np.pad(Y_in, [(0, 0), (L, L)], mode='edge')  # (M, N+2*L)

    grad_X = np.pad(np.diff(X_in_padded, 1, axis=-1), [(0, 0), (0, 1)], mode='edge')  # (M, N+2*L)
    grad_G = np.pad(np.diff(G_in_padded, 1, axis=-1), [(0, 0), (0, 1)], mode='edge')  # (M, N+2*L)

    K_TI_hor = np.empty_like(X_in)
    K_hor = np.empty_like(X_in)
    X_max = np.empty_like(X_in)
    X_min = np.empty_like(X_in)
    M, N = X_in.shape

    from tqdm import tqdm
    # for j in range(N):
    for j in tqdm(range(N)):
        ###### extract windows
        X_in_L = X_in_padded[:, j:j + LL]  # (M, 2L+1)
        G_in_L = G_in_padded[:, j:j + LL]  # (M, 2L+1)
        Y_in_L = Y_in_padded[:, j:j + LL]  # (M, 2L+1)
        grad_X_L = grad_X[:, j:j + LL]  # (M, 2L+1)
        grad_G_L = grad_G[:, j:j + LL]  # (M, 2L+1)

        ###### Transient improvement filtering
        ## Compute min and max on windows (Eqs. (2) and (3))
        X_in_L_max, X_in_L_min = compute_local_max_and_min(X_in_L)  # (M,N) arrays  # TODO: recursive

        ## Main TI process (Eq. (1))
        mask = X_in_L[:, L] > G_in_L[:, L]
        mask_not = np.bitwise_not(mask)
        X_pf = rho[0] * X_in_L_max + rho[1] * X_in_L + rho[2] * X_in_L_min  # (M,N), init X_pf
        X_pf[mask_not] = rho[0] * X_in_L_min[mask_not] + rho[1] * X_in_L[mask_not] + rho[2] * X_in_L_max[mask_not]  # (M,N), replace

        ## Restricting the range of admissible values (Eq. (4))
        X_TImax = np.array(X_in_L)  # X_in_L == 0
        X_TImin = np.array(X_in_L)
        X_TImin[mask] = np.maximum(X_in_L_min[mask], G_in_L[mask])  # X_in_L > G_in_L
        X_TImax[mask_not] = np.minimum(X_in_L_max[mask_not], G_in_L[mask_not])  # X_in_L < G_in_L

        ## Clipping the filtered imaged in the admissible set on values (Eq. (5))
        X_TI = np.clip(X_pf, a_min=X_TImin, a_max=X_TImax)  # (M, 2L+1)
        K_TI_hor[:, j] = X_TI[:, L] - G_in_L[:, L]  # (M, 1)

        X_max[:, j] = np.squeeze(X_in_L_max, 1)
        X_min[:, j] = np.squeeze(X_in_L_min, 1)

        # RGB2KbKr conversion (Eq. (7))
        K_TI = X_TI - G_in_L   # (M, 2L+1)

        ##### False color filtering
        ## Chromaticity sign (Eq. (11))
        S_K = compute_S_K(K_TI, tau)  # (M, 2L+1)
        ## Gradients' weights (Eq. (12))
        w_K = compute_w_K(K_TI, grad_X=grad_X_L, grad_G=grad_G_L, Y=Y_in_L, alpha_X=alpha_X)  # (M, 2L+1)
        ## Combinations of the weights (Eq. (10))
        W_K = S_K * w_K  # (M, 2L+1)

        ## Linear filtering with clipping (Eqs. (9) and (10))
        W_K = W_K / np.sum(W_K, axis=-1, keepdims=True)  # (M, 2*L+1)
        K_hor[:, j] = np.einsum('ij,ij->i', W_K, clip(K_TI, K_TI[..., L]))  # (M, 1)

    return K_hor, K_TI_hor, X_max, X_min


def transiant_improvement(X_in, G_in, L_hor=7, L_ver=4, rho=(-0.25, 1.375, -0.125)):
    ## Horizontal pass
    X_TI_hor = transiant_improvement1D(X_in, G_in, L_hor, rho)
    ## Vertical pass
    X_TI_ver = transiant_improvement1D(X_in.T, G_in.T, L_ver, rho).T
    ## Full filtering
    X_TI = transiant_improvement1D(X_TI_hor.T, G_in.T, L_ver, rho).T

    return X_TI, X_TI_hor, X_TI_ver


def transiant_improvement1D(X_in, G_in, L, rho):
    M, N = X_in.shape
    LL = 2*L+1

    X_in_padded = np.pad(X_in, [(0,0), (L, L)], mode='edge')  # (M, N+2*L)
    G_in_padded = np.pad(G_in, [(0,0), (L, L)], mode='edge')  # (M, N+2*L)

    X_TI = np.empty_like(X_in)

    for j in range(N):
        X_in_L = X_in_padded[:, j:j+LL]
        G_in_L = G_in_padded[:, j:j+LL]

        ## Compute min and max on windows (Eqs. (2) and (3))
        X_max, X_min = compute_local_max_and_min(X_in_L)  # (M,N) arrays

        ## Main TI process (Eq. (1))
        mask = X_in_L[:, L] > G_in_L[:, L]
        mask_not = np.bitwise_not(mask)
        X_pf = rho[0] * X_max + rho[1] * X_in_L + rho[2] * X_min  # (M,N), init X_pf
        X_pf[mask_not] = rho[0] * X_min[mask_not] + rho[1] * X_in_L[mask_not] + rho[2] * X_max[mask_not]  # (M,N), replace

        ## Restricting the range of admissible values (Eq. (4))
        X_TImax = X_TImin = X_in_L  # X_in_L == 0
        X_TImin[mask] = np.maximum(X_min[mask], G_in_L[mask])   # X_in_L > G_in_L
        X_TImax[mask_not] = np.minimum(X_max[mask_not], G_in_L[mask_not])   # X_in_L < G_in_L

        ## Clipping the filtered imaged in the admissible set on values (Eq. (5))
        X_TI[:, j] = np.clip(X_pf, a_min=X_TImin, a_max=X_TImax)

    return X_TI


def compute_local_max_and_min(X):
    LL = X.shape[-1]  # (M, 2L+1)
    L = (LL-1) // 2

    ## compute min and max in two directions
    X_Emax = np.max(X[:, -L-1:], axis=-1, keepdims=True)  # (N,1)
    X_Emin = np.min(X[:, -L-1:], axis=-1, keepdims=True)  # (N,1)
    X_Wmax = np.max(X[:, :L+1], axis=-1, keepdims=True)  # (N,1)
    X_Wmin = np.min(X[:, :L+1], axis=-1, keepdims=True)  # (N,1)

    ## Select the best values
    X_max = X_Wmax
    X_min = X_Emin
    mask = X_Emax - X_Wmin >= X_Wmax - X_Emin
    X_max[mask] = X_Emax[mask]
    X_min[mask] = X_Wmin[mask]

    return X_max, X_min


def false_color_filtering(K, X, G, Y, alpha_X, L_hor=7, L_ver=4, tau=15./255):
    K_hor = false_color_filtering1D(K, X, G, Y, L_hor, tau, alpha_X)
    K_ver = false_color_filtering1D(K.T, X.T, G.T, Y.T, L_ver, tau, alpha_X).T

    return K_hor, K_ver


def false_color_filtering1D(K, X, G, Y, L, tau, alpha_X):
    N, M = K.shape

    LL = 2*L+1

    K_padded = np.pad(K, [(0, 0), (L, L)], mode='edge')  # (M, N+2*L+1)
    X_padded = np.pad(X, [(0, 0), (L, L)], mode='edge')  # (M, N+2*L+1)
    G_padded = np.pad(G, [(0, 0), (L, L)], mode='edge')  # (M, N+2*L+1)

    grad_X = np.pad(np.diff(X_padded, axis=-1), [(0, 0), (0, 1)], mode='edge')  # (M, N+2*L+1)
    grad_G = np.pad(np.diff(G_padded, axis=-1), [(0, 0), (0, 1)], mode='edge')  # (M, N+2*L+1)

    K_hor = np.empty_like(K)  # (M,N)

    ## Horizontal pass
    for j in range(M):
        K_L = K_padded[:, j:j+LL]
        S_K = compute_S_K(K_L, tau)  # (M, N+2*L+1)
        w_K = compute_w_K(K_L, grad_X[:, j:j+LL], grad_G[:, j:j+LL], Y[:, j:j+LL], alpha_X)  # (M, N+2*L+1)
        W = S_K * w_K  # (M, N+2*L+1)

        ## Filtering (Eq. (8))
        K_hor[:, j] = np.sum(W * clip(K_L, K_L[:, L]), axis=-1) / np.sum(W, axis=-1)  # (M)

    return K_hor


def compute_S_K(K, tau):
    LL = K.shape[-1]  # (M, 2L+1)
    L = (LL-1) // 2
    K_ref = K[:, L:L+1]  # (M, 1)
    ## Compute S_K (Eq. (11))
    S = np.empty_like(K)
    mask = np.bitwise_or(np.sign(K_ref) == np.sign(K), np.abs(K) < tau)
    S[mask] = 1
    return S


def compute_w_K(K, grad_X, grad_G, Y, alpha_X):
    LL = K.shape[-1]  # (M, 2L+1)
    L = (LL - 1) // 2
    Y_ref = Y[:, L:L + 1]  # (M, 1)
    ## Compute w_K (Eqs (12), (13), (14) and (15))
    ## Compute D_G (Eq. (13))
    D_G = np.abs(grad_G)
    ## Compute D_X (Eq. (14))
    D_X = np.maximum(np.abs(grad_X), alpha_X * np.abs(K))
    ## Compute D_Y (Eq. (15))
    D_Y = np.abs(Y_ref - Y)
    ## Compute w_K (Eq. (12))
    return 1.0 / (D_G + D_X + D_Y + 1e-8)


def clip(K, K_ref):
    ## Compute clip (Eq. (9))
    mask = K_ref > 0
    K[mask] = np.minimum(K[mask], K_ref[mask][:, None])
    mask = K_ref < 0
    K[mask] = np.maximum(K[mask], K_ref[mask][:, None])
    return K


def arbitration(K, K_TI, X, G, X_max, X_min, beta_X, L_hor, L_ver, gamma_1=0.5, gamma_2=0.25):
    M, N = X.shape

    L = L_hor
    LL = 2 * L + 1

    X_padded = np.pad(X, [(0, 0), (L, L)], mode='edge')
    G_padded = np.pad(G, [(0, 0), (L, L)], mode='edge')

    ## Compute the Contrast images (Eq. (20))
    X_contrast_hor = np.empty_like(X)  # (M, N), horizontal pass
    for j in range(N):
        X_L = X_padded[:, j:j+LL]  # (M, 2*L_hor+1)
        G_L = G_padded[:, j:j+LL]  # (N, 2*L_hor+1)
        Xp_max, Xp_min = compute_local_max_and_min_color_constraints(X_L, G_L, beta_X)  # (M,1)
        X_contrast_hor[:, j] = np.squeeze(Xp_max - Xp_min, 1)

    L = L_ver
    LL = 2 * L + 1

    X_padded = np.pad(X.T, [(0, 0), (L, L)], mode='edge')
    G_padded = np.pad(G.T, [(0, 0), (L, L)], mode='edge')

    X_contrast_ver = np.empty_like(X.T)  # (N, M), vertical pass
    for j in range(M):
        X_L = X_padded[:, j:j+LL]  # (N, 2*L_ver+1)
        G_L = G_padded[:, j:j+LL]  # (N, 2*L_ver+1)
        Xp_max, Xp_min = compute_local_max_and_min_color_constraints(X_L, G_L, beta_X)  # (M,1)
        X_contrast_ver[:, j] = np.squeeze(Xp_max - Xp_min, 1)
    X_contrast_ver = X_contrast_ver.T  # (M, N)

    ## Compute X_contrast (Eq. (21))
    X_contrast = X_contrast_ver
    mask = X_contrast_hor > X_contrast_ver
    X_contrast[mask] = X_contrast_hor[mask]

    ## Compute the arbitration weight (Eq. (22))
    alpha_K = compute_alpha_K(X_contrast, X_max=X_max, X_min=X_min, gamma_1=gamma_1, gamma_2=gamma_2, use_f2=False)

    ## Finally compute the final filtered image (Eq. (17))
    K_out = (1 - alpha_K) * K_TI + alpha_K * K

    return K_out


def compute_local_max_and_min_color_constraints(X, G, beta):
    ## Recompute updated local max and min (Eq .(19))
    LL = X.shape[-1]  # (M, 2L+1)
    L = (LL-1) // 2

    ## compute min and max in two directions
    X_Emax = np.max(X[:, -L-1:] - beta * np.abs(X[:, -L-1:] - G[:, -L-1:]), axis=-1, keepdims=True)  # (M,1)
    X_Emin = np.min(X[:, -L-1:] + beta * np.abs(X[:, -L-1:] - G[:, -L-1:]), axis=-1, keepdims=True)  # (M,1)
    X_Wmax = np.max(X[:, :L+1] - beta * np.abs(X[:, :L+1] - G[:, :L+1]), axis=-1, keepdims=True)  # (M,1)
    X_Wmin = np.min(X[:, :L+1] + beta * np.abs(X[:, :L+1] - G[:, :L+1]), axis=-1, keepdims=True)  # (M,1)

    ## Select the best values
    X_max = X_Wmax
    X_min = X_Emin
    mask = X_Emax - X_Wmin >= X_Wmax - X_Emin
    X_max[mask] = X_Emax[mask]
    X_min[mask] = X_Wmin[mask]

    return X_max, X_min


def compute_alpha_K(X, X_max=None, X_min=None, gamma_1=0.5, gamma_2=0.25, use_f2=False):
    ## Compute alpha_K using f_2 in the paper (Eqs. (22) and (23))
    num = np.maximum(X, 0)
    if use_f2:
        assert((X_max is not None) and (X_min is not None))
        denom = np.minimum(np.maximum(X_max - X_min, gamma_2), gamma_1)
    else:
        denom = gamma_1
    return np.minimum(num / denom, 1.0)


def KbKrtoRGB(J):
    R = J[..., 0] + J[..., 2]
    G = J[..., 0]
    B = J[..., 0] + J[..., 1]
    return np.stack([R, G, B], axis=-1)


def RGBtoKbKr(I):
    G = I[..., 1]
    Kb = I[..., 2] - G
    Kr = I[..., 0] - G
    return np.stack([G, Kb, Kr], axis=-1)
