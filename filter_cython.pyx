import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. Therefore we recommend
# always calling "import_array" whenever you "cimport numpy"
np.import_array()

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.floatc

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float32_t DTYPE_t


def chromatic_removal(np.ndarray[DTYPE_t, ndim=3] I_in, unsigned int L_hor=7, unsigned int L_ver=4,
                      np.ndarray[DTYPE_t, ndim=1] rho=np.array((-0.25, 1.375, -0.125)),
                      float tau=15./255, float gamma_1=0.5, float gamma_2=0.25):
    assert I_in.dtype == DTYPE

    ## Introduction
    cdef np.ndarray[DTYPE_t, ndim=2] R_in = I_in[..., 0]
    cdef np.ndarray[DTYPE_t, ndim=2] G_in = I_in[..., 1]
    cdef np.ndarray[DTYPE_t, ndim=2] B_in = I_in[..., 2]
    cdef np.ndarray[DTYPE_t, ndim=2] Y_in = 0.299 * R_in + 0.587 * G_in + 0.114 * B_in

    ## Filtering
    # Horizontal pass
    cdef np.ndarray[DTYPE_t, ndim=2] K_r_hor, K_rTI_hor, R_max_hor, R_min_hor
    cdef np.ndarray[DTYPE_t, ndim=2] K_b_hor, K_bTI_hor, B_max_hor, B_min_hor
    K_r_hor, K_rTI_hor, R_max_hor, R_min_hor = ti_and_ca_filtering1D(R_in, G_in, Y_in, L_hor, rho=rho,
                                                                     tau=tau, alpha_X=0.5)
    K_b_hor, K_bTI_hor, B_max_hor, B_min_hor = ti_and_ca_filtering1D(B_in, G_in, Y_in, L_hor, rho=rho,
                                                                     tau=tau, alpha_X=1.0)

    # Vertical pass
    cdef np.ndarray[DTYPE_t, ndim=2] K_r_ver, K_rTI_ver, R_max_ver, R_min_ver
    cdef np.ndarray[DTYPE_t, ndim=2] K_b_ver, K_bTI_ver, B_max_ver, B_min_ver
    K_r_ver, K_rTI_ver, R_max_ver, R_min_ver = ti_and_ca_filtering1D(R_in.T, G_in.T, Y_in.T, L_ver, rho=rho,
                                                                     tau=tau, alpha_X=0.5)
    K_b_ver, K_bTI_ver, B_max_ver, B_min_ver = ti_and_ca_filtering1D(B_in.T, G_in.T, Y_in.T, L_ver, rho=rho,
                                                                     tau=tau, alpha_X=1.0)
    K_r_ver = K_r_ver.T
    K_b_ver = K_b_ver.T
    K_rTI_ver = K_rTI_ver.T
    K_bTI_ver = K_bTI_ver.T
    R_max_ver = R_max_ver.T
    B_max_ver = B_max_ver.T
    R_min_ver = R_min_ver.T
    B_min_ver = B_min_ver.T

    ## Arbitration
    cdef np.ndarray mask
    # Build the 2D images from the vertically and the horizontally FC filtered images (Eqs. (16))
    # Kb
    cdef np.ndarray[DTYPE_t, ndim=2] K_b
    K_b = K_b_ver
    mask = np.abs(K_b_hor) < np.abs(K_b_ver)
    K_b[mask] = K_b_hor[mask]
    #Kr
    cdef np.ndarray[DTYPE_t, ndim=2] K_r
    K_r = K_r_ver
    mask = np.abs(K_r_hor) < np.abs(K_r_ver)
    K_r[mask] = K_r_hor[mask]

    # Build the 2D images from the vertically and the horizontally TI filtered images (Eqs. (18))
    # Kb
    cdef np.ndarray[DTYPE_t, ndim=2] K_bTI
    cdef np.ndarray[DTYPE_t, ndim=2] B_max
    cdef np.ndarray[DTYPE_t, ndim=2] B_min
    K_bTI = K_bTI_ver
    B_max = B_max_ver
    B_min = B_min_ver
    mask = np.abs(K_bTI_hor) < np.abs(K_bTI_ver)
    K_bTI[mask] = K_bTI_hor[mask]
    B_max[mask] = B_max_hor[mask]
    B_min[mask] = B_min_hor[mask]

    # Kr
    cdef np.ndarray[DTYPE_t, ndim=2] K_rTI
    cdef np.ndarray[DTYPE_t, ndim=2] R_max
    cdef np.ndarray[DTYPE_t, ndim=2] R_min
    K_rTI = K_rTI_ver
    R_max = R_max_ver
    R_min = R_min_ver
    mask = np.abs(K_rTI_hor) < np.abs(K_rTI_ver)
    K_rTI[mask] = K_rTI_hor[mask]
    R_max[mask] = R_max_hor[mask]
    R_min[mask] = R_min_hor[mask]

    # Contrast arbitration
    cdef np.ndarray[DTYPE_t, ndim=2] K_rout
    cdef np.ndarray[DTYPE_t, ndim=2] K_bout
    K_rout = arbitration(K_r, K_rTI, R_in, G_in, R_max, R_min, beta_X=1.0, L_hor=L_hor, L_ver=L_ver,
                         gamma_1=gamma_1, gamma_2=gamma_2)
    K_bout = arbitration(K_b, K_bTI, B_in, G_in, B_max, B_min, beta_X=0.25, L_hor=L_hor, L_ver=L_ver,
                         gamma_1=gamma_1, gamma_2=gamma_2)

    # Final RGB conversion (Eq. (24))
    return np.stack([K_rout + G_in, G_in, K_bout + G_in], axis=-1)


def ti_and_ca_filtering1D(np.ndarray[DTYPE_t, ndim=2] X_in, np.ndarray[DTYPE_t, ndim=2] G_in,
                          np.ndarray[DTYPE_t, ndim=2] Y_in, int L, np.ndarray[DTYPE_t, ndim=1] rho,
                          float alpha_X, float tau):
    cdef int LL = 2 * L + 1
    cdef np.ndarray[DTYPE_t, ndim=2] X_in_padded = np.pad(X_in, [(0, 0), (L, L)], mode='edge')  # (M, N+2*L)
    cdef np.ndarray[DTYPE_t, ndim=2] G_in_padded = np.pad(G_in, [(0, 0), (L, L)], mode='edge')  # (M, N+2*L)
    cdef np.ndarray[DTYPE_t, ndim=2] Y_in_padded = np.pad(Y_in, [(0, 0), (L, L)], mode='edge')  # (M, N+2*L)

    cdef np.ndarray[DTYPE_t, ndim=2] grad_X = np.pad(np.diff(X_in_padded, 1, axis=-1), [(0, 0), (0, 1)],
                                                     mode='edge')  # (M, N+2*L)
    cdef np.ndarray[DTYPE_t, ndim=2] grad_G = np.pad(np.diff(G_in_padded, 1, axis=-1), [(0, 0), (0, 1)],
                                                     mode='edge')  # (M, N+2*L)

    cdef np.ndarray[DTYPE_t, ndim=2] K_TI_hor = np.zeros_like(X_in)
    cdef np.ndarray[DTYPE_t, ndim=2] K_hor = np.zeros_like(X_in)
    cdef np.ndarray[DTYPE_t, ndim=2] X_max = np.zeros_like(X_in)
    cdef np.ndarray[DTYPE_t, ndim=2] X_min = np.zeros_like(X_in)
    cdef int M, N
    M = X_in.shape[0]
    N = X_in.shape[1]

    cdef np.ndarray X_in_L, G_in_L, Y_in_L, grad_X_L, grad_G_L, X_in_L_max, X_in_L_min, mask, mask_not
    cdef np.ndarray X_pf, X_TImax, X_TImin, X_TI, K_TI, S_K, w_K, W_K
    for j in range(N):
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


def compute_local_max_and_min(np.ndarray[DTYPE_t, ndim=2] X):
    cdef int LL = X.shape[-1]  # (M, 2L+1)
    cdef int L = (LL-1) // 2

    ## compute min and max in two directions
    cdef np.ndarray[DTYPE_t, ndim=2] X_Emax = np.max(X[:, -L-1:], axis=-1, keepdims=True)  # (N,1)
    cdef np.ndarray[DTYPE_t, ndim=2] X_Emin = np.min(X[:, -L-1:], axis=-1, keepdims=True)  # (N,1)
    cdef np.ndarray[DTYPE_t, ndim=2] X_Wmax = np.max(X[:, :L+1], axis=-1, keepdims=True)  # (N,1)
    cdef np.ndarray[DTYPE_t, ndim=2] X_Wmin = np.min(X[:, :L+1], axis=-1, keepdims=True)  # (N,1)

    ## Select the best values
    cdef np.ndarray[DTYPE_t, ndim=2] X_max = X_Wmax
    cdef np.ndarray[DTYPE_t, ndim=2] X_min = X_Emin
    cdef np.ndarray mask = X_Emax - X_Wmin >= X_Wmax - X_Emin
    X_max[mask] = X_Emax[mask]
    X_min[mask] = X_Wmin[mask]

    return X_max, X_min


def compute_S_K(np.ndarray[DTYPE_t, ndim=2] K, float tau):
    cdef int LL = K.shape[-1]  # (M, 2L+1)
    cdef int L = (LL-1) // 2
    cdef np.ndarray[DTYPE_t, ndim=2] K_ref = K[:, L:L+1]  # (M, 1)
    ## Compute S_K (Eq. (11))
    cdef np.ndarray[DTYPE_t, ndim=2] S = np.zeros_like(K)
    cdef np.ndarray mask = np.bitwise_or(np.sign(K_ref) == np.sign(K), np.abs(K) < tau)
    S[mask] = 1
    return S


def compute_w_K(np.ndarray[DTYPE_t, ndim=2] K, np.ndarray[DTYPE_t, ndim=2] grad_X, np.ndarray[DTYPE_t, ndim=2] grad_G,
                np.ndarray[DTYPE_t, ndim=2] Y, float alpha_X):
    cdef int LL = K.shape[-1]  # (M, 2L+1)
    cdef int L = (LL - 1) // 2
    cdef np.ndarray[DTYPE_t, ndim=2] Y_ref = Y[:, L:L + 1]  # (M, 1)
    ## Compute w_K (Eqs (12), (13), (14) and (15))
    ## Compute D_G (Eq. (13))
    cdef np.ndarray[DTYPE_t, ndim=2] D_G = np.abs(grad_G)
    ## Compute D_X (Eq. (14))
    cdef np.ndarray[DTYPE_t, ndim=2] D_X = np.maximum(np.abs(grad_X), alpha_X * np.abs(K))
    ## Compute D_Y (Eq. (15))
    cdef np.ndarray[DTYPE_t, ndim=2] D_Y = np.abs(Y_ref - Y)
    ## Compute w_K (Eq. (12))
    return 1.0 / (D_G + D_X + D_Y + 1e-8)


def clip(np.ndarray[DTYPE_t, ndim=2] K, np.ndarray[DTYPE_t, ndim=2] K_ref):
    ## Compute clip (Eq. (9))
    cdef np.ndarray mask = K_ref > 0
    K[mask] = np.minimum(K[mask], K_ref[mask][:, None])
    mask = K_ref < 0
    K[mask] = np.maximum(K[mask], K_ref[mask][:, None])
    return K


def arbitration(np.ndarray[DTYPE_t, ndim=2] K, np.ndarray[DTYPE_t, ndim=2] K_TI, np.ndarray[DTYPE_t, ndim=2] X,
                np.ndarray[DTYPE_t, ndim=2] G, np.ndarray[DTYPE_t, ndim=2] X_max, np.ndarray[DTYPE_t, ndim=2] X_min,
                float beta_X, int L_hor, int L_ver, float gamma_1=0.5, float gamma_2=0.25):
    cdef int M, N
    M = X.shape[0]
    N = X.shape[1]

    cdef int L = L_hor
    cdef int LL = 2 * L + 1

    cdef np.ndarray[DTYPE_t, ndim=2] X_padded = np.pad(X, [(0, 0), (L, L)], mode='edge')
    cdef np.ndarray[DTYPE_t, ndim=2] G_padded = np.pad(G, [(0, 0), (L, L)], mode='edge')

    ## Compute the Contrast images (Eq. (20))
    cdef np.ndarray[DTYPE_t, ndim=2] X_contrast_hor = np.zeros_like(X)  # (M, N), horizontal pass
    cdef np.ndarray[DTYPE_t, ndim=2] X_L, G_L, Xp_max, Xp_min
    for j in range(N):
        X_L = X_padded[:, j:j+LL]  # (M, 2*L_hor+1)
        G_L = G_padded[:, j:j+LL]  # (N, 2*L_hor+1)
        Xp_max, Xp_min = compute_local_max_and_min_color_constraints(X_L, G_L, beta_X)  # (M,1)
        X_contrast_hor[:, j] = np.squeeze(Xp_max - Xp_min, 1)

    L = L_ver
    LL = 2 * L + 1

    X_padded = np.pad(X.T, [(0, 0), (L, L)], mode='edge')
    G_padded = np.pad(G.T, [(0, 0), (L, L)], mode='edge')

    cdef np.ndarray[DTYPE_t, ndim=2] X_contrast_ver = np.zeros_like(X.T)  # (N, M), vertical pass
    for j in range(M):
        X_L = X_padded[:, j:j+LL]  # (N, 2*L_ver+1)
        G_L = G_padded[:, j:j+LL]  # (N, 2*L_ver+1)
        Xp_max, Xp_min = compute_local_max_and_min_color_constraints(X_L, G_L, beta_X)  # (M,1)
        X_contrast_ver[:, j] = np.squeeze(Xp_max - Xp_min, 1)
    X_contrast_ver = X_contrast_ver.T  # (M, N)

    ## Compute X_contrast (Eq. (21))
    cdef np.ndarray[DTYPE_t, ndim=2] X_contrast = X_contrast_ver
    cdef np.ndarray mask = X_contrast_hor > X_contrast_ver
    X_contrast[mask] = X_contrast_hor[mask]

    ## Compute the arbitration weight (Eq. (22))
    cdef np.ndarray[DTYPE_t, ndim=2] alpha_K = compute_alpha_K(X_contrast, X_max=X_max, X_min=X_min,
                                              gamma_1=gamma_1, gamma_2=gamma_2)

    ## Finally compute the final filtered image (Eq. (17))
    cdef np.ndarray[DTYPE_t, ndim=2] K_out = (1 - alpha_K) * K_TI + alpha_K * K

    return K_out


def compute_local_max_and_min_color_constraints(np.ndarray X, np.ndarray G, float beta):
    ## Recompute updated local max and min (Eq .(19))
    cdef int LL = X.shape[-1]  # (M, 2L+1)
    cdef int L = (LL-1) // 2

    ## compute min and max in two directions
    cdef np.ndarray[DTYPE_t, ndim=2] X_Emax = np.max(X[:, -L-1:] - beta * np.abs(X[:, -L-1:] - G[:, -L-1:]),
                                                     axis=-1, keepdims=True)  # (M,1)
    cdef np.ndarray[DTYPE_t, ndim=2] X_Emin = np.min(X[:, -L-1:] + beta * np.abs(X[:, -L-1:] - G[:, -L-1:]),
                                                     axis=-1, keepdims=True)  # (M,1)
    cdef np.ndarray[DTYPE_t, ndim=2] X_Wmax = np.max(X[:, :L+1] - beta * np.abs(X[:, :L+1] - G[:, :L+1]),
                                                     axis=-1, keepdims=True)  # (M,1)
    cdef np.ndarray[DTYPE_t, ndim=2] X_Wmin = np.min(X[:, :L+1] + beta * np.abs(X[:, :L+1] - G[:, :L+1]),
                                                     axis=-1, keepdims=True)  # (M,1)

    ## Select the best values
    cdef np.ndarray[DTYPE_t, ndim=2] X_max = X_Wmax
    cdef np.ndarray[DTYPE_t, ndim=2] X_min = X_Emin
    cdef np.ndarray mask = X_Emax - X_Wmin >= X_Wmax - X_Emin
    X_max[mask] = X_Emax[mask]
    X_min[mask] = X_Wmin[mask]

    return X_max, X_min


def compute_alpha_K(np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] X_max=None,
                    np.ndarray[DTYPE_t, ndim=2] X_min=None, float gamma_1=0.5, float gamma_2=0.25):
    ## Compute alpha_K using f_2 in the paper (Eqs. (22) and (23))
    cdef np.ndarray[DTYPE_t, ndim=2] num = np.maximum(X, 0)
    cdef np.ndarray[DTYPE_t, ndim=2] denom
    denom = np.minimum(np.maximum(X_max - X_min, gamma_2), gamma_1)
    return np.minimum(num / denom, 1.0)
