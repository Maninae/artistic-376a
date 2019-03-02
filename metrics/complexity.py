from PIL import Image
from functools import reduce
import math, operator
import os

def compress_image(image, quality=10):
	im1 = Image.open(image+'.jpg')
	IMAGE_10 = os.path.join(image+'_10.jpg')
	im1.save(IMAGE_10,"JPEG", quality=quality)
	im10 = Image.open(IMAGE_10)
	return image+'_10'

def rms(image1, image2):
	h1 = Image.open(image1+'.jpg').histogram()
	h2 = Image.open(image2+'.jpg').histogram()

	rms = math.sqrt(reduce(operator.add,
		map(lambda a,b: (a-b)**2, h1, h2))/len(h1))

	return rms

def compression_ratio(image1, image2):
	b1 = os.path.getsize(image1+'.jpg')	
	b2 = os.path.getsize(image2+'.jpg')	
	return b1/b2

def get_complexity(image, quality=10):
	image2 = compress_image(image)
	rms_ = rms(image, image2)
	cr = compression_ratio(image, image2)
	return rms_ / cr

image = '../images/mona_lisa'
print(get_complexity(image))
