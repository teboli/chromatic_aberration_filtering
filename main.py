import numpy as np
from skimage import io, transform, img_as_float32, img_as_ubyte
import matplotlib.pyplot as plt
import time
import tqdm

# import filter
import filter_cython as filter
# import filter_recursive as filter


# impath = 'facade_blurry.jpg'
impath = 'bridge_blurry.jpg'
# img = io.imread(impath)
img = img_as_float32(io.imread(impath))
# img = transform.rescale(img, scale=(0.25, 0.25, 1))
# img = transform.rescale(img, scale=(0.5, 0.5, 1))


rho = np.array([-0.25, 1.375, -0.125], dtype=img.dtype)
L_hor = 2
L_ver = 2
alpha_R = 0.5
alpha_B = 1.0
beta_R = 1.0
beta_B = 0.25
tau = 15. / 255
gamma_1 = 128. / 255
gamma_2 = 64. / 255

tic = time.time()
impred = filter.chromatic_removal(img, L_hor, L_ver, rho, tau, alpha_R, alpha_B, beta_R, beta_B, 
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
