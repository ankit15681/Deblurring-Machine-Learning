import tensorflow as tf
from PIL import Image, ImageEnhance
import numpy as np
sess = tf.InteractiveSession()
avg_ssim = 0
for i in range(100):
    deblurred = Image.open("C:/Users/ankit/Desktop/btp/predicted/" + str(i) + ".png")
    target = Image.open("C:/Users/ankit/Desktop/btp/target/" + str(i) + ".png")
    blurred = Image.open("C:/Users/ankit/Desktop/btp/blurred/" + str(i) + ".png")

    deblurred = np.array(deblurred)
    target = np.array(target)
    blurred = np.array(blurred)

    ssim = tf.image.psnr(tf.convert_to_tensor(target, dtype=tf.float32),
                          tf.convert_to_tensor(deblurred, dtype=tf.float32), max_val=255)

    avg_ssim += ssim

    print(i)
    print("SSIM: " + str(ssim.eval()))
    print("\n")

avg_ssim = avg_ssim.eval()
print(avg_ssim / 100)
sess.close()
