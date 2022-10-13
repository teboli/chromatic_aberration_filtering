import numpy as np
from skimage import io, img_as_float32, img_as_ubyte
import matplotlib.pyplot as plt
import time

from filter_cython import chromatic_removal


impath = 'images/bridge_blurry.jpg'
# impath = 'images/facade_blurry.jpg'
img = img_as_float32(io.imread(impath))


rho = np.array([-0.25, 1.375, -0.125], dtype=img.dtype)
L_hor = 14
L_ver = 4
alpha_R = 0.5
alpha_B = 1.0
beta_R = 1.0
beta_B = 0.25
tau = 15. / 255
gamma_1 = 128. / 255
gamma_2 = 64. / 255

print('Use the parameters:')
print('  rho:    ', list(rho))
print('  L_ver:  ', L_ver)
print('  L_hor:  ', L_hor)
print('  alpha_R:', alpha_R)
print('  alpha_B:', alpha_B)
print('  beta_R: ', beta_R)
print('  beta_B: ', beta_B)
print('  gamma_1:', gamma_1)
print('  gamma_2:', gamma_2)
print('  tau:', tau)


print('Start restoration...')
tic = time.time()
impred = chromatic_removal(img, L_hor, L_ver, rho, tau, alpha_R, alpha_B, beta_R, beta_B, 
                                  gamma_1, gamma_2)
toc = time.time()
print("Elapsed time: %2.2f sec" % (toc - tic))

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(impred)
plt.show()

print(impred.max(), impred.min())


io.imsave(impath.split('_')[0] + '_prediction.png', img_as_ubyte(impred))

print('Done!')
