
import skimage.io
import skimage.measure

img_a = skimage.io.imread('/home/DISK2/SR_NN_SPC/code/SR_SPC/experiment_44/validate/generate_7290000.png', as_grey=True)
img_b = skimage.io.imread('/home/DISK2/SR_NN_SPC/code/SR_SPC/experiment_44/validate/hr_7290000.png', as_grey=True)

img_ssim = skimage.measure.compare_ssim(img_a, img_b)
img_psnr = skimage.measure.compare_psnr(img_a, img_b)

