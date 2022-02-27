from __future__ import print_function
import tensorflow as tf
import binNetworks as net
import logging,os
import numpy as np
from scipy import misc
from skimage import filters
import cv2
import matplotlib.pyplot as plt

imgh = 256
imgw = 256

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("logs_dir", "train_model_fusion_11k-log/", "path to writer logs directory")
tf.flags.DEFINE_string("dataset", "image_test2019/", "path to writer logs directory")
tf.flags.DEFINE_string("imgtype", "png", "path to writer logs directory")
tf.flags.DEFINE_float("overlap", "0.1", "overlap of patches")
tf.flags.DEFINE_bool("multiscale", "False", "whether use multiscale")
tf.flags.DEFINE_integer("num_block", "6", "the number of blocks")

overlap = 4.0


def imshow(img):
	#img = misc.toimage(img, cmin=0, cmax=255)
	plt.imshow(img,cmap='gray')
	#print 'get max value in image is:',np.max(img)
	plt.show()


def imshowlist(imglist):
	#img = misc.toimage(img, cmin=0, cmax=255)
	nImg = len(imglist)
	fig = plt.figure()
	for n in range(nImg):
		fig.add_subplot(1,nImg,n)
		plt.imshow(imglist[n],cmap='gray')

	#print 'get max value in image is:',np.max(img)
	plt.show()

def imsave(name, arr):
	"""Save an array to an image file.
	"""
	#im = misc.toimage(arr)
	im = misc.toimage(arr, cmin=0, cmax=255)
	im.save(name)
	return


def get_image_patch_multiscale(image,imgh,imgw,nimgh,nimgw,overlap=0.1):
	overlap_wid = int(imgw * overlap)
	overlap_hig = int(imgh * overlap)

	height,width = image.shape

	image_list = []
	posit_list = []

	for ys in range(0,height-nimgh,overlap_hig):
		ye = ys + imgh
		for xs in range(0,width-nimgw,overlap_wid):
			xe = xs + nimgw
			imgpath = image[ys:ye,xs:xe]

			imgpath = misc.imresize(imgpath,(imgh,imgw),interp='bicubic')
			image_list.append(imgpath)
			pos = np.array([ys,xs])
			posit_list.append(pos)

	# last coloum
	for xs in range(0,width-nimgw,overlap_wid):
		xe = xs + nimgw
		ye = height
		ys = ye - nimgh
		imgpath = image[ys:ye,xs:xe]

		imgpath = misc.imresize(imgpath,(imgh,imgw),interp='bicubic')
		image_list.append(imgpath)
		pos = np.array([ys,xs])
		posit_list.append(pos)

	# last row
	for ys in range(0,height-nimgw,overlap_hig):
		ye = ys + nimgh
		xe = width
		xs = xe - nimgh
		imgpath = image[ys:ye,xs:xe]

		imgpath = misc.imresize(imgpath,(imgh,imgw),interp='bicubic')
		image_list.append(imgpath)
		pos = np.array([ys,xs])
		posit_list.append(pos)

	# last rectangle
	ye = height
	ys = ye - nimgh
	xe = width
	xs = xe - nimgw
	imgpath = image[ys:ye,xs:xe]

	imgpath = misc.imresize(imgpath,(imgh,imgw),interp='bicubic')
	image_list.append(imgpath)
	pos = np.array([ys,xs])
	posit_list.append(pos)
	return np.stack(image_list),posit_list


def get_image_patch(image,imgh,imgw,reshape=None,overlap=0.1):

	overlap_wid = int(imgw * overlap)
	overlap_hig = int(imgh * overlap)

	height,width = image.shape

	image_list = []
	posit_list = []

	for ys in range(0,height-imgh,overlap_hig):
		ye = ys + imgh
		if ye > height:
			ye = height
		for xs in range(0,width-imgw,overlap_wid):
			xe = xs + imgw
			if xe > width:
				xe = width
			imgpath = image[ys:ye,xs:xe]
			if reshape is not None:
				imgpath = cv2.resize(imgpath.astype('float'), dsize=reshape)
			image_list.append(imgpath)
			pos = np.array([ys,xs,ye,xe])
			posit_list.append(pos)

	# last coloum
	for xs in range(0,width-imgw,overlap_wid):
		xe = xs + imgw
		if xe > width:
			xe = width
		ye = height
		ys = ye - imgh
		if ys < 0:
			ys = 0

		imgpath = image[ys:ye,xs:xe]
		if reshape is not None:
			imgpath = cv2.resize(imgpath.astype('float'), dsize=reshape)
		image_list.append(imgpath)
		pos = np.array([ys,xs,ye,xe])
		posit_list.append(pos)

	# last row
	for ys in range(0,height-imgh,overlap_hig):
		ye = ys + imgh
		if ye > height:
			ye = height
		xe = width
		xs = xe - imgw
		if xs < 0:
			xs = 0

		imgpath = image[ys:ye,xs:xe]
		if reshape is not None:
			imgpath = cv2.resize(imgpath.astype('float'), dsize=reshape)
		image_list.append(imgpath)
		pos = np.array([ys,xs,ye,xe])
		posit_list.append(pos)

	# last rectangle
	ye = height
	ys = ye - imgh
	if ys < 0:
		ys = 0
	xe = width
	xs = xe - imgw
	if xs < 0:
		xs = 0

	imgpath = image[ys:ye,xs:xe]
	if reshape is not None:
		imgpath = cv2.resize(imgpath.astype('float'), dsize=reshape)
	image_list.append(imgpath)
	pos = np.array([ys,xs,ye,xe])
	posit_list.append(pos)

	#return np.stack(image_list),posit_list
	return image_list,posit_list


def refinement(patch):
	men = np.mean(patch)
	std = np.std(patch)

	imglist=[]
	#thres = men - 0.2 * std
	thres = men * (1+0.2*((std/128.0)-1))
	res_tmp = patch <= thres
	#imglist.append(res_tmp)
	#imglist.append(patch)

	if np.sum(res_tmp) > 0:
		#print('rescale')
		patch = misc.imresize(patch,patch.shape,interp='bicubic')
	else:
		patch = (1-res_tmp)*255

	#imglist.append(patch)
	#imshowlist(imglist)
	return patch

def local_thres(patch):
	men = np.mean(patch)
	std = np.std(patch)
	thres = men * (1+0.2*((std/128.0)-1))
	#thres = filters.threshold_otsu(patch)
	mask = patch < thres
	#mask = mask * 255
	return mask

def main(argv=None):

	image = tf.placeholder(tf.float32,shape=[None,imgh,imgw,1],name='image')
	imbin = tf.placeholder(tf.float32,shape=[None,imgh,imgw,1],name='imbin')
	is_training = tf.placeholder(tf.bool,name='is_training')

	num_block = 6
	nlayers =0
	overlap = FLAGS.overlap
	multiscale = FLAGS.multiscale
	num_block = FLAGS.num_block

	bin_pred_list = net.buildnet(image,num_block,nlayers)
	model_saver = tf.train.Saver()

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	model_logs_dir = FLAGS.logs_dir
	print('-'*20)
	print(model_logs_dir)
	ckpt = tf.train.get_checkpoint_state(model_logs_dir)
	if ckpt and ckpt.model_checkpoint_path:
		model_saver.restore(sess,ckpt.model_checkpoint_path)
		print('*'*20)
		print("Model restored...")
		print('*'*20)

	#image_test_dir = '/mantis/PaperWork/binary/dataset/test/'+FLAGS.dataset+'/'
	image_test_dir = FLAGS.dataset


	imagetype = FLAGS.imgtype
	if multiscale:
		scalelist = [0.75,1.0,1.25,1.5]
	else:
		scalelist = [1.0]

	for root,sub,images in os.walk(image_test_dir):
		for img in images:
			if not img.endswith(imagetype):
				continue

			if img.startswith('GT'):
				continue

			#if not img.startswith('PR06'):
			#	continue

			print('processing the image:', img)

			image_test = misc.imread(image_test_dir + img,mode='L')
			oh,ow = image_test.shape


			res_out = np.zeros((oh,ow))
			num_hit = np.zeros((oh,ow))

			for s in range(num_block):

				#if s <2:
				#	continue

				print('%d-iter is running!'%s)



				for scale in scalelist:

					crpW = int(scale*imgw)
					crpH = int(scale*imgh)

					reshape = (imgw,imgh)
					image_patch,poslist = get_image_patch(image_test,crpH,crpW,reshape,overlap=overlap)
					#image_patch += rpatch
					#poslist += rpos

					image_patch = np.stack(image_patch)
					image_patch = np.expand_dims(image_patch,axis=3)

					print('scale: %f get patches: %d'%(scale,len(poslist)))

					npath = len(poslist)

					batch_size = 10

					nstep = int( npath / batch_size ) + 1

					for ns in range(nstep):
						ps = ns * batch_size
						pe = ps + batch_size
						if pe >= npath:
							pe = npath

						pathes = image_patch[ps:pe]
						if pathes.shape[0] == 0:
							continue

						feed_dict = {image:pathes,is_training:False}
						pred_bin = sess.run(bin_pred_list[s],feed_dict=feed_dict)

						pred_bin = np.squeeze(pred_bin)


						#print('pred_bin shape:',pred_bin.shape)

						if ns == 0:
							pred_bin_list = pred_bin
						else:
							#print('ndim is:',pred_bin.ndim,pred_bin.shape)
							if pred_bin.ndim < 3:
								pred_bin = np.expand_dims(pred_bin,axis=0)
							#print('ndim is:',pred_bin.ndim,pred_bin.shape)
							pred_bin_list = np.concatenate((pred_bin_list,pred_bin),axis=0)

					print(pred_bin_list.shape,npath,nstep,'*'*20)
					for n in range(npath):
						ys = poslist[n][0]
						xs = poslist[n][1]
						ye = poslist[n][2]
						xe = poslist[n][3]

						reH = ye - ys
						reW = xe - xs


						if npath == 1:
							resPath = pred_bin_list
						else:
							resPath = pred_bin_list[n]

						resPath = refinement(resPath)

						resPath = cv2.resize(resPath.astype('float'), dsize=(reW,reH))

						num_hit[ys:ye,xs:xe] += 1
						res_out[ys:ye,xs:xe] += resPath

			res_out = res_out / num_hit
			imsave('pred-'+img[:-4]+'-list-'+str(s)+'.png',res_out)


if __name__ == '__main__':
	tf.app.run(main=main)
