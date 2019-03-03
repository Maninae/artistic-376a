import numpy as np 
import matplotlib.pyplot as plt 
from scipy.mics import imread, imresize

IMG_DIR="./../datasets"

def load_dataset(load_dir=IMG_DIR):
	"""
	Loads the dataset. 
	"""
	####################################################################
	# Todo
	pass
	####################################################################

def display_img(im, edge_sigma_1, edge_sigma3):
	"""
	Displays images on 3x1 subplot with different levels of edge filtering.

	@param im: original image
	@param edge_sigma_1: edges of original image with Canny filtering (Gaussian sigma = 1)
	@param edge_sigma_3: edges of original image with Canny filtering (Gaussian sigma = 3)
	"""

	fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)

	ax1.imshow(im, cmap=plt.cm.gray)
	ax1.axis('off')
	ax1.set_title('noisy image', fontsize=20)

	ax2.imshow(edge_sigma_1, cmap=plt.cm.gray)
	ax2.axis('off')
	ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

	ax3.imshow(edge_sigma3, cmap=plt.cm.gray)
	ax3.axis('off')
	ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

	fig.tight_layout()

	plt.show()

def content_quantity_measure(im, im_edges, threshold=0.3):
	"""
	Calculates the amount of information in edges of the image. 
	Zeros out elements below the threshold, and return number of non-zero elements.
	"""
	####################################################################
	# Todo
	pass
	####################################################################

if __name__ == "__main__":

	# Loads the dataset
	dataset = load_dataset()

	# Runs Canny Edge detection on 2 different levels.
	im_edges_1 = feature.canny(im)
	im_edges_3 = feature.canny(im, sigma=3)

	# Displays examples of images with edge extracted.
	display_img(im, im_edges_1, im_edges_3)
