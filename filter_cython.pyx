import numpy as np
cimport cython

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

from cython.parallel import prange
from libc.math cimport fmin, fmax, fabs

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. Therefore we recommend
# always calling "import_array" whenever you "cimport numpy"
np.import_array()

DTYPE = np.float32

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float32_t DTYPE_t
ctypedef np.uint8_t uint8


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef DTYPE_t[:, ::1] transpose(DTYPE_t[:, ::1] X):
    cdef Py_ssize_t M = X.shape[0]
    cdef Py_ssize_t N = X.shape[1]
    cdef Py_ssize_t i, j
    cdef DTYPE_t[:, ::1] Xt = np.zeros((N, M), dtype=DTYPE)

    for i in range(0, M, 1):
        for j in range(0, N, 1):
            Xt[j, i] = X[i, j]
    return Xt


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def chromatic_removal(np.ndarray[DTYPE_t, ndim=3] I_in, int L_hor, int L_ver,
                      np.ndarray[DTYPE_t, ndim=1] rho, DTYPE_t tau, DTYPE_t alpha_R, DTYPE_t alpha_B, DTYPE_t beta_R,
                      DTYPE_t beta_B, DTYPE_t gamma_1, DTYPE_t gamma_2):
    cdef Py_ssize_t M, N, i, j
    M = I_in.shape[0]
    N = I_in.shape[1]

    ## Introduction
    cdef DTYPE_t[:, ::1] R_in = np.zeros((M, N), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] G_in = np.zeros((M, N), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] B_in = np.zeros((M, N), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] Y_in = np.zeros((M, N), dtype=DTYPE)
    for i in range(0, M, 1):
    # for i in prange(M, nogil=True):
        for j in range(0, N, 1):
            R_in[i, j] = I_in[i, j, 0]
            G_in[i, j] = I_in[i, j, 1]
            B_in[i, j] = I_in[i, j, 2]
            Y_in[i, j] = 0.299 * R_in[i, j] + 0.587 * G_in[i, j] + 0.114 * B_in[i, j]

    ## Filtering
    # Horizontal pass
    cdef DTYPE_t[:, ::1] K_r_hor   = np.zeros((M, N), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] K_rTI_hor = np.zeros((M, N), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] R_max_hor = np.zeros((M, N), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] R_min_hor = np.zeros((M, N), dtype=DTYPE)
    K_r_hor, K_rTI_hor, R_max_hor, R_min_hor = ti_and_ca_filtering1D(R_in, G_in, Y_in, L_hor, rho, tau, alpha_R)
    # ti_and_ca_filtering1D(R_in, G_in, Y_in, L_hor, rho, tau, alpha_R, K_r_hor, K_rTI_hor, R_max_hor, R_min_hor)
    cdef DTYPE_t[:, ::1] K_b_hor   = np.zeros((M, N), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] K_bTI_hor = np.zeros((M, N), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] B_max_hor = np.zeros((M, N), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] B_min_hor = np.zeros((M, N), dtype=DTYPE)
    K_b_hor, K_bTI_hor, B_max_hor, B_min_hor = ti_and_ca_filtering1D(B_in, G_in, Y_in, L_hor, rho, tau, alpha_B)
    # ti_and_ca_filtering1D(B_in, G_in, Y_in, L_hor, rho, tau, alpha_B, K_b_hor, K_bTI_hor, B_max_hor, B_min_hor)
    #
    # # Vertical pass
    cdef DTYPE_t[:, ::1] K_r_ver   = np.zeros((N, M), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] K_rTI_ver = np.zeros((N, M), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] R_max_ver = np.zeros((N, M), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] R_min_ver = np.zeros((N, M), dtype=DTYPE)
    K_r_ver, K_rTI_ver, R_max_ver, R_min_ver = ti_and_ca_filtering1D(transpose(R_in), transpose(G_in), transpose(Y_in),
                                                                     L_ver, rho, tau, alpha_R)
    # ti_and_ca_filtering1D(transpose(R_in), transpose(G_in), transpose(Y_in), L_ver, rho, tau, alpha_R,
    #                       K_r_ver, K_rTI_ver, R_max_ver, R_min_ver)
    cdef DTYPE_t[:, ::1] K_b_ver   = np.zeros((N, M), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] K_bTI_ver = np.zeros((N, M), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] B_max_ver = np.zeros((N, M), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] B_min_ver = np.zeros((N, M), dtype=DTYPE)
    K_b_ver, K_bTI_ver, B_max_ver, B_min_ver = ti_and_ca_filtering1D(transpose(B_in), transpose(G_in), transpose(Y_in),
                                                                     L_ver, rho, tau, alpha_B)
    # ti_and_ca_filtering1D(transpose(B_in), transpose(G_in), transpose(Y_in), L_ver, rho, tau, alpha_B,
    #                       K_b_ver, K_bTI_ver, B_max_ver, B_min_ver)

    K_r_ver = transpose(K_r_ver)
    K_b_ver = transpose(K_b_ver)
    K_rTI_ver = transpose(K_rTI_ver)
    K_bTI_ver = transpose(K_bTI_ver)
    R_max_ver = transpose(R_max_ver)
    B_max_ver = transpose(B_max_ver)
    R_min_ver = transpose(R_min_ver)
    B_min_ver = transpose(B_min_ver)

    print('max(K_rTI_hor)', np.max(K_rTI_hor))
    print('max(K_bTI_hor)', np.max(K_bTI_hor))

    print('max(K_rTI_ver)', np.max(K_rTI_ver))
    print('max(K_bTI_ver)', np.max(K_bTI_ver))

    ## Arbitration
    # Build the 2D images from the vertically and the horizontally FC filtered images (Eqs. (16))
    # Kb
    cdef DTYPE_t[:, ::1] K_b = K_b_ver
    for i in range(0, M, 1):
    # for i in prange(M, nogil=True):
        for j in range(0, N, 1):
            if fabs(K_b_hor[i, j]) < fabs(K_b_ver[i, j]):
                K_b[i, j] = K_b_hor[i, j]

    #Kr
    cdef DTYPE_t[:, ::1] K_r = K_r_ver
    for i in range(0, M, 1):
    # for i in prange(M, nogil=True):
        for j in range(0, N, 1):
            if fabs(K_r_hor[i, j]) < fabs(K_r_ver[i, j]):
                K_r[i, j] = K_r_hor[i, j]

    # Build the 2D images from the vertically and the horizontally TI filtered images (Eqs. (18))
    # Kb
    cdef DTYPE_t[:, ::1] K_bTI = K_bTI_ver
    cdef DTYPE_t[:, ::1] B_max = B_max_ver
    cdef DTYPE_t[:, ::1] B_min = B_min_ver
    for i in range(0, M, 1):
    # for i in prange(M, nogil=True):
        for j in range(0, N, 1):
            if fabs(K_bTI_hor[i, j]) < fabs(K_bTI_ver[i, j]):
                K_bTI[i, j] = K_bTI_hor[i, j]
                B_max[i, j] = B_max_hor[i, j]
                B_min[i, j] = B_min_hor[i, j]

    # Kr
    cdef DTYPE_t[:, ::1] K_rTI = K_rTI_ver
    cdef DTYPE_t[:, ::1] R_max = R_max_ver
    cdef DTYPE_t[:, ::1] R_min = R_min_ver
    for i in range(0, M, 1):
    # for i in prange(M, nogil=True):
        for j in range(0, N, 1):
            if fabs(K_rTI_hor[i, j]) < fabs(K_rTI_ver[i, j]):
                K_rTI[i, j] = K_rTI_hor[i, j]
                R_max[i, j] = R_max_hor[i, j]
                R_min[i, j] = R_min_hor[i, j]

    print('max(K_rTI)', np.max(K_rTI))
    print('max(K_bTI)', np.max(K_bTI))

    print('max(K_r)', np.max(K_r))
    print('max(K_b)', np.max(K_b))

    # Contrast arbitration
    cdef DTYPE_t[:, ::1] K_rout = arbitration(K_r, K_rTI, R_in, G_in, R_max, R_min, beta_R, L_hor, L_ver, gamma_1, gamma_2)
    cdef DTYPE_t[:, ::1] K_bout = arbitration(K_b, K_bTI, B_in, G_in, B_max, B_min, beta_B, L_hor, L_ver, gamma_1, gamma_2)

    K_rout = K_rTI
    K_bout = K_bTI

    # Final RGB conversion (Eq. (24))
    cdef DTYPE_t[:, :, ::1] I_out = np.zeros((M, N, 3), dtype=DTYPE)
    for i in range(0, M, 1):
    # for i in prange(M, nogil=True):
        for j in range(0, N, 1):
            I_out[i, j, 0] = K_rout[i, j] + G_in[i, j]
            I_out[i, j, 1] = G_in[i, j]
            I_out[i, j, 2] = K_bout[i, j] + G_in[i, j]

    print('max(K_rout)', np.max(K_rout))
    print('max(K_bout)', np.max(K_bout))

    return np.clip(np.asarray(I_out), 0, 1)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef DTYPE_t[:, ::1] grad(DTYPE_t[:, ::1] X, Py_ssize_t M, Py_ssize_t N):
    cdef DTYPE_t[:, ::1] grad_X = np.zeros((M, N), dtype=DTYPE)
    cdef Py_ssize_t i, j
    for i in range(0, M, 1):
    # for i in prange(start=0, stop=M, nogil=True):
        for j in range(1, N, 1):
            grad_X[i, j] = X[i, j] - X[i, j - 1]
    return grad_X


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int sign(DTYPE_t input) nogil:
    if input > 0:
        return 1
    elif input < 0:
        return -1
    else:
        return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef DTYPE_t clip(DTYPE_t K, DTYPE_t K_ref) nogil:
    ## Compute clip (Eq. (9))
    if K > 0:
        K = fmin(K, K_ref)
    elif K < 0:
        K = fmax(K, K_ref)
    else:
        K = K_ref
    return K


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def ti_and_ca_filtering1D(DTYPE_t[:, ::1] X_in, DTYPE_t[:, ::1] G_in, DTYPE_t[:, ::1] Y_in, int L,
                                DTYPE_t[::1] rho, DTYPE_t alpha_X, DTYPE_t tau):
# cdef void ti_and_ca_filtering1D(DTYPE_t[:, ::1] X_in, DTYPE_t[:, ::1] G_in, DTYPE_t[:, ::1] Y_in, unsigned int L,
#                                 DTYPE_t[::1] rho, DTYPE_t alpha_X, DTYPE_t tau, DTYPE_t[:, ::1] K_hor,
#                                 DTYPE_t[:, ::1] K_Ti_hor, DTYPE_t[:, ::1] X_max, DTYPE_t[:, ::1] X_min):

    cdef Py_ssize_t M, N, i, j, l
    # cdef Py_ssize_t l = 0
    cdef unsigned int LL = 2 * L + 1
    cdef DTYPE_t eps = 1e-8

    M = X_in.shape[0]
    N = X_in.shape[1]

    cdef DTYPE_t[:, ::1] K_hor   = np.zeros((M, N), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] K_TI_hor = np.zeros((M, N), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] X_max = np.zeros((M, N), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] X_min = np.zeros((M, N), dtype=DTYPE)

    ## Compute the horizontal gradients
    cdef DTYPE_t[:, ::1] grad_X = grad(X_in, M, N)
    cdef DTYPE_t[:, ::1] grad_G = grad(G_in, M, N)

    cdef DTYPE_t[:, ::1] X_TImax = np.zeros((M, LL), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] X_TImin = np.zeros((M,LL), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] X_TI = np.zeros((M, LL), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] K_TI = np.zeros((M, LL), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] W_K = np.zeros((M, LL), dtype=DTYPE)
    cdef DTYPE_t[::1] W_K_sum = np.zeros(M, dtype=DTYPE)
    cdef DTYPE_t[::1] X_Emax = np.zeros(M, dtype=DTYPE)
    cdef DTYPE_t[::1] X_Emin = np.zeros(M, dtype=DTYPE)
    cdef DTYPE_t[::1] X_Wmax = np.zeros(M, dtype=DTYPE)
    cdef DTYPE_t[::1] X_Wmin = np.zeros(M, dtype=DTYPE)
    cdef DTYPE_t[::1] X_in_L_max = np.zeros(M, dtype=DTYPE)
    cdef DTYPE_t[::1] X_in_L_min = np.zeros(M, dtype=DTYPE)

    for i in range(L, M - L, 1):
    # for i in prange(L, M - L, 1, nogil=True):
        for j in range(L, N - L, 1):
            W_K_sum[i] = 0
            X_Emax[i] = 0
            X_Emin[i] = 0
            X_Wmax[i] = 0
            X_Wmin[i] = 0

            ###### Transient improvement filtering
            ## Compute min and max on windows (Eq. (2))
            for l in range(0, L + 1, 1):
                if X_in[i, j + l] > X_Emax[i]:
                    X_Emax[i] = X_in[i, j + l]
                if X_in[i, j + l] < X_Emin[i]:
                    X_Emin[i] = X_in[i, j + l]
                if X_in[i, j - L + l] > X_Wmax[i]:
                    X_Wmax[i] = X_in[i, j - L + l]
                if X_in[i, j - L + l] < X_Wmin[i]:
                    X_Wmin[i] = X_in[i, j - L + l]

            ## Select the best couple of values (Eq. (3))
            if (X_Emax[i] - X_Wmin[i]) >= (X_Wmax[i] - X_Emin[i]):
                X_max[i, j] = X_Emax[i]
                X_min[i, j] = X_Wmin[i]
            else:
                X_max[i, j] = X_Wmax[i]
                X_min[i, j] = X_Emin[i]

            # for l in prange(-L, L + 1, 1, nogil=True):
            for l in range(-L, L+1, 1):
                ## Main TI process (Eq. (1)) and Eq. (4)) (we use X_TI for X_pf as they are the same thing)
                if X_in[i, j] > G_in[i, j]:
                    X_TI[i, l] = rho[0] * X_max[i, j] + rho[1] * X_in[i, j + l] + rho[2] * X_min[i, j]
                    X_TImax[i, l] = X_in[i, j + l]
                    X_TImin[i, l] = fmax(X_in_L_min[i], G_in[i, j + l])
                else:
                    X_TI[i, l] = rho[0] * X_min[i, j] + rho[1] * X_in[i, j + l] + rho[2] * X_max[i, j]
                    X_TImax[i, l] = fmin(X_in_L_max[i], G_in[i, j + l])
                    X_TImin[i, l] = X_in[i, j + l]

                ## Clipping the filtered imaged in the admissible set on values (Eq. (5))
                if X_TI[i, l] > X_TImax[i, l]:
                    X_TI[i, l] = X_TImax[i, l]
                elif X_TI[i, l] < X_TImin[i, l]:
                    X_TI[i, l] = X_TImin[i, l]

                # RGB2KbKr conversion (Eq. (7))
                K_TI[i, l] = X_TI[i, l] - G_in[i, j + l]   # (2L+1)

                ##### False color filtering
                ## Computing W_K in Eq. (10)
                ## Chromaticity sign (Eq. (11))
                if (sign(K_TI[i, L]) == sign(K_TI[i, l])) or (fabs(K_TI[i, j + l]) < tau):
                    W_K[i, l] = 1
                else:
                    W_K[i, l] = 0

                ## Gradients' weights (Eq. (12))
                W_K[i, l] = W_K[i, l] / (fabs(grad_G[i, j + l]) + fmax(fabs(grad_X[i, j + l]), alpha_X * fabs(K_TI[i, l])) + fabs(Y_in[i, j] - Y_in[i, j + l])  + eps)

                ## Linear filtering with clipping (Eqs. (9) and (10))
                K_hor[i, j] = K_hor[i, j] + W_K[i, l] * clip(K_TI[i, l], K_TI[i, L])
                W_K_sum[i] = W_K_sum[i] + W_K[i, l]

            ## Linear filtering (Eqs. (9))
            K_hor[i, j] = K_hor[i, j] / (W_K_sum[i] + eps)

            # if i == L and j == L:
            #     print('X_TI', np.max(X_TI[i]))
            #     print('G_TI', np.max(G_in[i]))
            #     print('K_TI', np.max(K_TI[i]))
            #     print('W_Kij', np.max(W_K[i]))
    print('X_in', np.max(X_in))
    print('grad_X', np.max(grad_X))
    print('X_max', np.max(X_max))
    print('X_in_L_max', np.max(X_in_L_max))
    print('W_K', np.max(W_K))
    print('W_K_sum', np.max(W_K_sum))
    print('K_hor', np.max(K_hor))

    return K_hor, K_TI_hor, X_max, X_min

#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# cdef (DTYPE_t, DTYPE_t) compute_local_max_and_min(DTYPE_t[:] X, DTYPE_t X_Emax, DTYPE_t X_Emin, DTYPE_t X_Wmax,
#                                                   DTYPE_t X_Wmin, int L, Py_ssize_t l):
#     ## X has shape (2*L+1)
#     ## compute min and max in two directions
#     for l in range(0, L+1, 1):
#         if X[L + l] > X_Emax:
#             X_Emax = X[L + l]
#         if X[L + l] < X_Emin:
#             X_Emin = X[L + l]
#         if X[l] > X_Wmax:
#             X_Wmax = X[l]
#         if X[l] < X_Emin:
#             X_Wmin = X[l]
#
#     ## Select the best couple of values
#     if (X_Emax - X_Wmin) >= (X_Wmax - X_Emin):
#         X_max = X_Emax
#         X_min = X_Wmin
#     else:
#         X_max = X_Wmax
#         X_min = X_Emin
#
#     return X_max, X_min



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef DTYPE_t[:, ::1] arbitration(DTYPE_t[:, ::1] K, DTYPE_t[:, ::1] K_TI, DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] G, DTYPE_t[:, ::1] X_max,
                DTYPE_t[:, ::1] X_min, DTYPE_t beta_X, int L_hor, int L_ver, DTYPE_t gamma_1,
                DTYPE_t gamma_2):
    cdef Py_ssize_t M, N, i, j
    cdef int l = 0
    M = X.shape[0]
    N = X.shape[1]

    cdef int L = L_hor

    ## Compute the Contrast images (Eq. (20))
    cdef DTYPE_t[:, ::1] X_contrast_hor = np.zeros((M, N), dtype=DTYPE)  # (M, N), horizontal pass
    cdef DTYPE_t[::1] Xp_max_hor = np.zeros(M, dtype=DTYPE)
    cdef DTYPE_t[::1] Xp_min_hor = np.zeros(M, dtype=DTYPE)

    cdef DTYPE_t[::1] X_Emax_hor = np.zeros(M, dtype=DTYPE)
    cdef DTYPE_t[::1] X_Emin_hor = np.zeros(M, dtype=DTYPE)
    cdef DTYPE_t[::1] X_Wmax_hor = np.zeros(M, dtype=DTYPE)
    cdef DTYPE_t[::1] X_Wmin_hor = np.zeros(M, dtype=DTYPE)

    for i in range(L, M - L, 1):
    # for i in prange(start=L, stop=M-L, nogil=True):
        for j in range(L, N - L, 1):
            Xp_max_hor[i], Xp_min_hor[i] = compute_local_max_and_min_color_constraints(X[i, j - L:j + L + 1],
                                                                                       G[i, j - L:j + L + 1],
                                                                                       X_Emax_hor[i], X_Emin_hor[i],
                                                                                       X_Wmax_hor[i], X_Wmin_hor[i],
                                                                                       beta_X, L, l)  # (1), (1)

            X_contrast_hor[i, j] = Xp_max_hor[i] - Xp_min_hor[i]

    L = L_ver

    cdef DTYPE_t[:, ::1] X_contrast_ver = np.zeros((N, M), dtype=DTYPE)  # (N, M), vertical pass
    cdef DTYPE_t[::1] Xp_max_ver = np.zeros(N, dtype=DTYPE)
    cdef DTYPE_t[::1] Xp_min_ver = np.zeros(N, dtype=DTYPE)

    cdef DTYPE_t[::1] X_Emax_ver = np.zeros(N, dtype=DTYPE)
    cdef DTYPE_t[::1] X_Emin_ver = np.zeros(N, dtype=DTYPE)
    cdef DTYPE_t[::1] X_Wmax_ver = np.zeros(N, dtype=DTYPE)
    cdef DTYPE_t[::1] X_Wmin_ver = np.zeros(N, dtype=DTYPE)

    cdef DTYPE_t[:, ::1] Xt = transpose(X)
    cdef DTYPE_t[:, ::1] Gt = transpose(G)

    for i in range(L, N - L, 1):
    # for i in prange(start=L, stop=N-L, nogil=True):
        for j in range(L, M - L, 1):
            Xp_max_ver[i], Xp_min_ver[i] = compute_local_max_and_min_color_constraints(Xt[i, j - L:j + L + 1],
                                                                                       Gt[i, j - L:j + L + 1],
                                                                                       X_Emax_ver[i], X_Emin_ver[i],
                                                                                       X_Wmax_ver[i], X_Wmin_ver[i],
                                                                                       beta_X, L, l)
            # ## Recompute updated local max and min (Eq .(19))
            # for l in range(0, L + 1, 1):
            #     if (X[L + l] - beta_X * fabs(X[L + l] - G[L + l])) > X_Emax_ver[i]:
            #         X_Emax_ver[i] = X[L + l] - beta_X * fabs(X[L + l] - G[L + l])
            #     if (X[L + l] + beta_X * fabs(X[L + l] - G[L + l])) < X_Emin_ver[i]:
            #         X_Emin_ver[i] = X[L + l] + beta_X * fabs(X[L + l] - G[L + l])
            #     if (X[l] - beta_X * fabs(X[l] - G[l])) > X_Wmax_ver[i]:
            #         X_Wmax_ver[i] = X[l] - beta_X * fabs(X[l] - G[l])
            #     if (X[l] + beta_X * fabs(X[l] - G[l])) < X_Wmin_ver[i]:
            #         X_Wmin_ver[i] = X[l] + beta_X * fabs(X[l] - G[l])
            #
            # ## Select the best couple of values
            # if (X_Emax[i] - X_Wmin[i]) >= (X_Wmax[i] - X_Emin[i]):
            #     Xp_max_ver[i] = X_Emax[i]
            #     Xp_min_ver[i] = X_Wmin[i]
            # else:
            #     Xp_max_ver[i] = X_Wmax[i]
            #     Xp_min_ver[i] = X_Emin[i]
            X_contrast_ver[i, j] = Xp_max_ver[i] - Xp_min_ver[i]
    X_contrast_ver = transpose(X_contrast_ver)  # (M, N)

    ## Compute X_contrast (Eq. (21))
    cdef DTYPE_t[:, ::1] X_contrast = np.zeros((M, N), dtype=DTYPE)
    for i in range(0, M, 1):
    # for i in prange(N, nogil=True):
        for j in range(0, N, 1):
            if X_contrast_hor[i, j] > X_contrast_ver[i, j]:
                X_contrast[i, j] = X_contrast_hor[i, j]
            else:
                X_contrast[i, j] = X_contrast_ver[i, j]

    ## Compute the arbitration weight (Eq. (22)) and the final filtered image (Eq. (17))
    cdef DTYPE_t alpha_K
    cdef DTYPE_t[:, ::1] K_out = np.zeros((M, N), dtype=DTYPE)
    for i in range(0, N, 1):
    # for i in prange(N, nogil=True):
        for j in range(0, M, 1):
            alpha_K = fmin(fmax(X[i, j], 0.0) / fmin(fmax(X_max[i, j] - X_min[i, j], gamma_2), gamma_1), 1.0)
            # K_out[i, j] = (1 - alpha_K) * K_TI[i, j] + alpha_K * K[i, j]
            K_out[i, j] = K_TI[i, j] + alpha_K * (K[i, j] - K_TI[i, j])

    return K_out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef (DTYPE_t, DTYPE_t) compute_local_max_and_min_color_constraints(DTYPE_t[:] X, DTYPE_t[:] G, DTYPE_t X_Emax,
                                                                    DTYPE_t X_Emin, DTYPE_t X_Wmax, DTYPE_t X_Wmin,
                                                                    DTYPE_t beta, int L, int l):
    ## Recompute updated local max and min (Eq .(19))
    for l in range(0, L + 1, 1):
        if (X[L + l] - beta * fabs(X[L + l] - G[L + l])) > X_Emax:
            X_Emax = X[L + l] - beta * fabs(X[L + l] - G[L + l])
        if (X[L + l] + beta * fabs(X[L + l] - G[L + l])) < X_Emin:
            X_Emin = X[L + l] + beta * fabs(X[L + l] - G[L + l])
        if (X[l] - beta * fabs(X[l] - G[l])) > X_Wmax:
            X_Wmax = X[l] - beta * fabs(X[l] - G[l])
        if (X[l] + beta * fabs(X[l] - G[l])) < X_Emin:
            X_Wmin = X[l] + beta * fabs(X[l] - G[l])

    ## Select the best couple of values
    if (X_Emax - X_Wmin) >= (X_Wmax - X_Emin):
        X_max = X_Emax
        X_min = X_Wmin
    else:
        X_max = X_Wmax
        X_min = X_Emin
    return X_max, X_min


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef DTYPE_t[:, ::1] compute_alpha_K(DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] X_max, DTYPE_t[:, ::1] X_min, DTYPE_t gamma_1,
                          DTYPE_t gamma_2):
    ## Compute alpha_K using f_2 in the paper (Eqs. (22) and (23))
    cdef Py_ssize_t i, j, N, M
    N = X.shape[0]
    M = X.shape[1]
    cdef DTYPE_t[:, ::1] out = np.zeros((M, N), dtype=DTYPE)

    for i in range(0, N, 1):
    # for i in prange(N, nogil=True):
        for j in range(0, M, 1):
            out[i, j] = fmin(fmax(X[i, j], 0.0) / fmin(fmax(X_max[i, j] - X_min[i, j], gamma_2), gamma_1), 1.0)
    return out
