import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from PIL import Image
from skimage import io, img_as_float, img_as_uint, color
from skimage.measure import structural_similarity as ssim

def svd_greyscale(img, k):
	#Compresses the given image by choosing the top k elements in SVD
	U, singular_vals, V = linalg.svd(img)
	rank = len(singular_vals)
	print "Image rank %r" % rank
	if k > rank:
		print "k is larger than rank of image %r" % rank
		return img
	# take columns less than k from U
	U_p = U[:,:k]
	# take rows less than k from V
	V_p = V[:k,:]
	# build the new S matrix with top k diagonal elements
	S_p = np.zeros((k, k), img.dtype)
	for i in range(k):
		S_p[i][i] = singular_vals[i]
	print "U_p shape "+str(np.shape(U_p))+" S_p shape "+str(np.shape(S_p))+" V_p shape "+str(np.shape(V_p))

	#compressed = np.dot(np.dot(U_p, S_p), V_p)
	compressed=rev_svd(U_p,S_p,V_p)
	ss = ssim(img, compressed,dynamic_range=compressed.max()-compressed.min())
	print "Structural similarity: %r" % ss
	return compressed

def svd_rgb(img_name, k_r, k_g, k_b,out_name):
	#Uses the svd_greyscale function for each of its colors
	img=io.imread(img_name)
	img=img_as_float(img)
	red_ch= svd_greyscale(img[:,:,0], k_r)
	green_ch = svd_greyscale(img[:,:,1], k_g)
	blue_ch = svd_greyscale(img[:,:,2], k_b)
	new_img = np.zeros(img.shape, img.dtype)
	rows = img.shape[0]
	cols = img.shape[1]
	depth = img.shape[2]
	for i in range(rows):
		for j in range(cols):
			for c in range(depth):
				val = 0
				if c == 0:
					val = red_ch[i][j]
				elif c == 1:
					val = green_ch[i][j]
				else:
					val = blue_ch[i][j]
				# float64 values must be between -1.0 and 1.0
				if val < -1.0:
					val = -1.0
				elif val > 1.0:
					val = 1.0
				new_img[i][j][c] = val
	io.imsave(out_name,new_img)
	return new_img

def compress_ratio(img_name, k):
	#The compression ratio in terms of input size and k
	img=io.imread(img_name)
	m = float(img.shape[0])
	n = float(img.shape[1])
	new_size = 0
	if len(img.shape) > 2:
		new_size += k[0] * (m + n + 1)
		new_size += k[1] * (m + n + 1)
		new_size += k[2] * (m + n + 1)
		return new_size / (3 * m * n)
	else:
		new_size = k[0] * (m + n + 1)
		return new_size / (m * n)

def rev_svd(U, S, V): #reverse of SVD.. 
	return np.dot(np.dot(U, S), V)

def svd_ssim(img_name, out_name,target_ss=0.9):
	img=io.imread(img_name)
	img=color.rgb2gray(img)
	img=img_as_float(img)
	rank = min(img.shape[0], img.shape[1])

	lt = 1
	rt = rank
	
	prev_ss = 100
	
	k = 1
	compressed = None
	U, singular_vals, V = linalg.svd(img)
	# binary search starts
	while lt < rt:	
		k = (lt + rt) / 2
		S_p = np.zeros((k, k), img.dtype)
		for i in range(k):
			S_p[i][i] = singular_vals[i]
		compressed = rev_svd(U[:,:k], S_p, V[:k,:])
		ss = ssim(img, compressed,dynamic_range=compressed.max()-compressed.min())
		if abs(ss - target_ss) < abs(prev_ss - target_ss):
			prev_ss = ss
			if ss > target_ss:
				rt = k
			else:
				lt = k
		else:
			break
	# linear search for better results (can be ignored)
	if prev_ss < target_ss:
		while 1:
			S_p = np.zeros((k + 1, k + 1), img.dtype)
			for i in range(k + 1):
				S_p[i][i] = singular_vals[i]
			compressed = rev_svd(U[:,:k+1], S_p, V[:k+1,:])
			ss = ssim(img, compressed,dynamic_range=compressed.max()-compressed.min())
			if abs(ss - target_ss) < abs(prev_ss - target_ss):
				prev_ss = ss
				k += 1	
			else:
				break
	else:
		while 1:
			S_p = np.zeros((k - 1, k - 1), img.dtype)
			for i in range(k - 1):
				S_p[i][i] = singular_vals[i]
			compressed = rev_svd(U[:,:k-1], S_p, V[:k-1,:])
			ss = ssim(img, compressed,dynamic_range=compressed.max()-compressed.min())
			if abs(ss - target_ss) < abs(prev_ss - target_ss):
				prev_ss = ss
				k -= 1
			else:
				break	
	print "Best k found %r with ssim %r" % (k, prev_ss)
	svd_rgb(img_name, k, k, k,out_name)
	return compressed


#svd_ssim("test.jpg","output.jpg",target_ss=0.7)
