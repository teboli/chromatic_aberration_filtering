import numpy as np
from skimage import io, transform, img_as_float32, img_as_ubyte
import matplotlib.pyplot as plt
import time
import tqdm

# import filter
import filter_recursive as filter

tau = 15
gamma_1 = 128
gamma_2 = 64

# impath = 'facade_blurry.jpg'
impath = 'bridge_blurry.jpg'
# img = io.imread(impath)
img = img_as_float32(io.imread(impath)); tau /= 255; gamma_1 /= 255; gamma_2 /= 255
# img = transform.rescale(img, scale=(0.25, 0.25, 1))
# img = transform.rescale(img, scale=(0.5, 0.5, 1))

tic = time.time()
impred = filter.chromatic_removal(img, tau=tau, gamma_1=gamma_1, gamma_2=gamma_2)
toc = time.time()

print("Elapsed time: %2.2f sec" % (toc - tic))

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(impred)
plt.show()

io.imsave(impath.split('_')[0] + '_prediction.png', img_as_ubyte(impred))

print('Done!')
