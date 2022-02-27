""" 
	This code is used for our paper:
	binarization 
	
	Author: Sheng He (heshengxgd@gmail.com)
	Uni. of Groingen, the Netherlands
"""

import tensorflow as tf
import binLayers as layers

def leaky_relu(x):
	return tf.maximum(x,0.25*x)

def buildInputDesnet(inputs,numblock):
	netlist = []
	net = inputs
	for n in range(numblock):
		with tf.variable_scope('block'+str(n)):
			rnet = resUNetBlock(net)
			net = layers.concat(net,rnet,'inc'+str(n))
			netlist.append(rnet)
	return netlist

def buildfusionnet(inputs,numblock):
	netlist = []
	net = inputs
	for n in range(numblock):
		with tf.variable_scope('block'+str(n)):
			net = resUNetBlock(net)
			netlist.append(net)

	initw = 1./numblock

	for n in range(numblock):
		w = tf.get_variable('w'+str(n),[1],tf.float32,tf.constant_initializer(initw))
		wnet = tf.multiply(netlist[n],w)
		if n == 0:
			fnet = wnet
		else:
			fnet = layers.concat(fnet,wnet,'o-'+str(n))
		
	fnet = tf.expand_dims(tf.reduce_mean(fnet,axis=3),axis=3)

	netlist.append(fnet)
	return netlist

def buildfusionDesnet(inputs,numblock):
	netlist = []
	mid_netlist = []
	net = inputs
	for n in range(numblock):
		print('n-th:',n)
		with tf.variable_scope('block'+str(n)):
			rnet = resUNetBlock(net)
			print('in rnet shape:',rnet.get_shape())
			mid_netlist.append(rnet)
			
			if n == 0:
				net = rnet
				netlist.append(rnet)
			else:
				net = layers.concat(net,rnet,'inc'+str(n))
			#net = rnet
			
				num_channels = rnet.get_shape()[-1].value
				w = tf.get_variable('w'+str(n),[num_channels],tf.float32,tf.constant_initializer(1.0/num_channels))
				wnet = tf.multiply(rnet,w)
			
				wnet = tf.reduce_sum(wnet,axis=3,keep_dims=True)
				netlist.append(wnet)
			
			
	'''
	#initw = 1./numblock
	print('number of list:',len(netlist),numblock)
	with tf.variable_scope('fusion'):
		for n in xrange(numblock):
			num_channels = netlist[n].get_shape()[-1].value
			w = tf.get_variable('w'+str(n),[num_channels],tf.float32,tf.constant_initializer(1.0/num_channels))
			wnet = tf.multiply(netlist[n],w)
		
		
	fnet = tf.expand_dims(tf.reduce_mean(fnet,axis=3),axis=3)
	netlist.append(fnet)	
	'''


	return netlist,mid_netlist

def buildfusionDenselynet(inputs,numblock):
	netlist = []
	net = inputs
	for n in range(numblock):
		with tf.variable_scope('block'+str(n)):
			net = denseResUnetBlock(net)
			netlist.append(net)

	initw = 1./numblock

	for n in range(numblock):
		w = tf.get_variable('w'+str(n),[1],tf.float32,tf.constant_initializer(initw))
		wnet = tf.multiply(netlist[n],w)
		if n == 0:
			fnet = wnet
		else:
			fnet = layers.concat(fnet,wnet,'o-'+str(n))
	
	fnet = tf.reduce_mean(fnet,axis=3)
	print('finall map shape is:',fnet.get_shape().value)
	netlist.append(fnet)
	return netlist

def buildnet(inputs,numblock,nlayers=0):
	netlist = []
	net = inputs
	for n in range(numblock):
		with tf.variable_scope('block'+str(n)):
			net = resUNetBlock(net,nlayers)
			netlist.append(net)
	return netlist

def buildDesNet(inputs,numblock):
	netlist = []
	net = inputs
	for n in range(numblock):
		with tf.variable_scope('block'+str(n)):
			net = denseResUnetBlock(net)
			netlist.append(net)
	return netlist

def buildHEDUnet(inputs,numblock):
	netlist = []
	side_net_list = []
	net  = inputs
	for n in range(numblock):
		with tf.variable_scope('block'+str(n)):
			side_outlayer,net = resHEDUNetBlock(net,nlayers=0,shortCut=True)
			side_net_list.append(side_outlayer)
			netlist.append(net)
	return netlist,side_net_list

def denseResUnetBlock(inputs):
	
	#numFilter=[16,32,64,128,256]
	numFilter=[8,16,32,48,64]
	
	#256
	conv1 = layers.conv2d(inputs,[3,3],numFilter[0],'conv1',activation_fn=leaky_relu)
	pool1 = layers.max_pooling(conv1,[2,2],'pool1')
	
	#128
	#print('layer2 input shape:',pool1.get_shape())
	
	conv2 = layers.conv2d(pool1,[3,3],numFilter[1],'conv2',activation_fn=leaky_relu)
	pool2 = layers.max_pooling(conv2,[2,2],'pool2')
	
	# 64
	pool13 = layers.max_pooling(conv1,[4,4],'pool13')
	pool13 = layers.concat(pool2,pool13,'maxcon13')
	
	#print('layer3 input shape:',pool2.get_shape(),pool13.get_shape())
	
	conv3 = layers.conv2d(pool13,[3,3],numFilter[2],'conv3',activation_fn=leaky_relu)
	pool3 = layers.max_pooling(conv3,[2,2],'pool3')
	
	# 32
	pool14 = layers.max_pooling(conv1,[8,8],'pool14')
	pool24 = layers.max_pooling(conv2,[4,4],'pool14')
	
	pool14 = layers.concat(pool3,pool14,'maxcon14')
	pool14 = layers.concat(pool24,pool14,'maxcon24')
	
	#print('layer4 input shape:',pool3.get_shape(),pool14.get_shape())
	conv4 = layers.conv2d(pool14,[3,3],numFilter[3],'conv4',activation_fn=leaky_relu)
	pool4 = layers.max_pooling(conv4,[2,2],'pool4')
	
	#16
	pool15 = layers.max_pooling(conv1,[16,16],'pool15')
	pool25 = layers.max_pooling(conv2,[8,8],'pool25')
	pool35 = layers.max_pooling(conv3,[4,4],'pool35')
	
	pool15 = layers.concat(pool4,pool15,'maxcon15')
	pool15 = layers.concat(pool25,pool15,'maxcon25')
	pool15 = layers.concat(pool35,pool15,'maxcon35')
	
	#print('layer5 input shape:',pool4.get_shape(),pool15.get_shape())
	conv5 = layers.conv2d(pool15,[3,3],numFilter[4],'conv5',activation_fn=leaky_relu)
	pool5 = layers.max_pooling(conv5,[2,2],'pool5')
	# 8
	
	up_conv6 = layers.deconv_upsample(pool5,2,'upsample6',activation_fn=leaky_relu)
	up_conv6 = layers.concat(up_conv6,conv5,'cont6')
	up_conv6 = layers.conv2d(up_conv6,[3,3],numFilter[3],'conv6',activation_fn=leaky_relu)
	
	
	up_conv7 = layers.deconv_upsample(up_conv6,2,'upsample7',activation_fn=leaky_relu)
	up_conv7 = layers.concat(up_conv7,conv4,'cont7')
	up_conv7 = layers.conv2d(up_conv7,[3,3],numFilter[2],'conv7',activation_fn=leaky_relu)
	
	up_conv8 = layers.deconv_upsample(up_conv7,2,'upsample8',activation_fn=leaky_relu)
	up_conv68 = layers.deconv_upsample(up_conv6,4,'upsample68',activation_fn=leaky_relu)
	
	up_conv8 = layers.concat(up_conv8,conv3,'cont8')
	up_conv8 = layers.concat(up_conv8,up_conv68,'cont68')
	
	up_conv8 = layers.conv2d(up_conv8,[3,3],numFilter[1],'conv8',activation_fn=leaky_relu)
	
	up_conv9 = layers.deconv_upsample(up_conv8,2,'upsample9',activation_fn=leaky_relu)
	up_conv69 = layers.deconv_upsample(up_conv6,8,'upsample69',activation_fn=leaky_relu)
	up_conv79 = layers.deconv_upsample(up_conv7,4,'upsample79',activation_fn=leaky_relu)
	
	up_conv9 = layers.concat(up_conv9,conv2,'cont9')
	up_conv9 = layers.concat(up_conv9,up_conv69,'cont69')
	up_conv9 = layers.concat(up_conv9,up_conv79,'cont79')
	
	up_conv9 = layers.conv2d(up_conv9,[3,3],numFilter[0],'conv9',activation_fn=leaky_relu)
	
	
	up_conv10 = layers.deconv_upsample(up_conv9,2,'upsample10',activation_fn=leaky_relu)
	up_conv610 = layers.deconv_upsample(up_conv6,16,'upsample610',activation_fn=leaky_relu)
	up_conv710 = layers.deconv_upsample(up_conv7,8,'upsample710',activation_fn=leaky_relu)
	up_conv810 = layers.deconv_upsample(up_conv8,4,'upsample810',activation_fn=leaky_relu)
	
	up_conv10 = layers.concat(up_conv10,conv1,'cont10')
	up_conv10 = layers.concat(up_conv10,up_conv610,'cont610')
	up_conv10 = layers.concat(up_conv10,up_conv710,'cont710')
	up_conv10 = layers.concat(up_conv10,up_conv810,'cont810')
	
	outlayer = layers.conv2d(up_conv10,[3,3],1,'conv10',activation_fn=leaky_relu)
	
	outlayer = outlayer + inputs
	
	return outlayer
	
# This is the normal unet
def resUNetBlock(inputs,nlayers=1):
	#numFilter=[8,16,32,64,128]
	numFilter=[16,32,64,128,256]
	
	num_filters_in = inputs.get_shape()[-1].value
	
	conv1 = layers.conv2d(inputs,[3,3],numFilter[0],'conv1',activation_fn=leaky_relu)
	for n in range(nlayers):
		conv1 = layers.conv2d(conv1,[3,3],numFilter[0],'conv1-'+str(n),activation_fn=leaky_relu)
		
	pool1 = layers.max_pooling(conv1,[2,2],'pool1')
	
	conv2 = layers.conv2d(pool1,[3,3],numFilter[1],'conv2',activation_fn=leaky_relu)
	for n in range(nlayers):
		conv2 = layers.conv2d(conv2,[3,3],numFilter[1],'conv2-'+str(n),activation_fn=leaky_relu)
		
	pool2 = layers.max_pooling(conv2,[2,2],'pool2')
	
	conv3 = layers.conv2d(pool2,[3,3],numFilter[2],'conv3',activation_fn=leaky_relu)
	for n in range(nlayers):
		conv3 = layers.conv2d(conv3,[3,3],numFilter[2],'conv3-'+str(n),activation_fn=leaky_relu)
		
	pool3 = layers.max_pooling(conv3,[2,2],'pool3')
	
	conv4 = layers.conv2d(pool3,[3,3],numFilter[3],'conv4',activation_fn=leaky_relu)
	for n in range(nlayers):
		conv4 = layers.conv2d(conv4,[3,3],numFilter[3],'conv4-'+str(n),activation_fn=leaky_relu)
		
	pool4 = layers.max_pooling(conv4,[2,2],'pool4')
	
	conv5 = layers.conv2d(pool4,[3,3],numFilter[4],'conv5',activation_fn=leaky_relu)
	for n in range(nlayers):
		conv5 = layers.conv2d(conv5,[3,3],numFilter[4],'conv5-'+str(n),activation_fn=leaky_relu)
		
	pool5 = layers.max_pooling(conv5,[2,2],'pool5')
	
	up_conv6 = layers.deconv_upsample(pool5,2,'upsample6',activation_fn=leaky_relu)
	up_conv6 = layers.concat(up_conv6,conv5,'cont6')
	up_conv6 = layers.conv2d(up_conv6,[3,3],numFilter[3],'conv6',activation_fn=leaky_relu)
	for n in range(nlayers):
		up_conv6 = layers.conv2d(up_conv6,[3,3],numFilter[3],'conv6-'+str(n),activation_fn=leaky_relu)
	
	up_conv7 = layers.deconv_upsample(up_conv6,2,'upsample7',activation_fn=leaky_relu)
	up_conv7 = layers.concat(up_conv7,conv4,'cont7')
	up_conv7 = layers.conv2d(up_conv7,[3,3],numFilter[2],'conv7',activation_fn=leaky_relu)
	for n in range(nlayers):
		up_conv7 = layers.conv2d(up_conv7,[3,3],numFilter[2],'conv7-'+str(n),activation_fn=leaky_relu)	
	
	up_conv8 = layers.deconv_upsample(up_conv7,2,'upsample8',activation_fn=leaky_relu)
	up_conv8 = layers.concat(up_conv8,conv3,'cont8')
	up_conv8 = layers.conv2d(up_conv8,[3,3],numFilter[1],'conv8',activation_fn=leaky_relu)
	for n in range(nlayers):
		up_conv8 = layers.conv2d(up_conv8,[3,3],numFilter[1],'conv8-'+str(n),activation_fn=leaky_relu)
	

	up_conv9 = layers.deconv_upsample(up_conv8,2,'upsample9',activation_fn=leaky_relu)
	up_conv9 = layers.concat(up_conv9,conv2,'cont9')
	up_conv9 = layers.conv2d(up_conv9,[3,3],numFilter[0],'conv9',activation_fn=leaky_relu)
	for n in range(nlayers):
		up_conv9 = layers.conv2d(up_conv9,[3,3],numFilter[0],'conv9-'+str(n),activation_fn=leaky_relu)
	
	up_conv10 = layers.deconv_upsample(up_conv9,2,'upsample10',activation_fn=leaky_relu)
	up_conv10 = layers.concat(up_conv10,conv1,'cont10')
	
	
	
	outlayer = layers.conv2d(up_conv10,[3,3],num_filters_in,'conv10',activation_fn=None)
	#outlayer = layers.conv2d(up_conv10,[3,3],num_filters_in,'conv10',activation_fn=leaky_relu)
	
	outlayer = outlayer + inputs
	
	#outlayer = tf.reduce_mean(outlayer,axis=3,keep_dims=True)
	
	return outlayer
	
# This is the normal unet
def resHEDUNetBlock(inputs,nlayers=1,shortCut=False):
	#numFilter=[8,16,32,64,128]
	numFilter=[16,32,64,128,256]
	
	side_outputs = []
	num_filters_in = inputs.get_shape()[-1].value
	
	conv1 = layers.conv2d(inputs,[3,3],numFilter[0],'conv1',activation_fn=leaky_relu)
	for n in range(nlayers):
		conv1 = layers.conv2d(conv1,[3,3],numFilter[0],'conv1-'+str(n),activation_fn=leaky_relu)
		
	pool1 = layers.max_pooling(conv1,[2,2],'pool1')
	
	conv2 = layers.conv2d(pool1,[3,3],numFilter[1],'conv2',activation_fn=leaky_relu)
	for n in range(nlayers):
		conv2 = layers.conv2d(conv2,[3,3],numFilter[1],'conv2-'+str(n),activation_fn=leaky_relu)
		
	pool2 = layers.max_pooling(conv2,[2,2],'pool2')
	
	conv3 = layers.conv2d(pool2,[3,3],numFilter[2],'conv3',activation_fn=leaky_relu)
	for n in range(nlayers):
		conv3 = layers.conv2d(conv3,[3,3],numFilter[2],'conv3-'+str(n),activation_fn=leaky_relu)
		
	pool3 = layers.max_pooling(conv3,[2,2],'pool3')
	
	conv4 = layers.conv2d(pool3,[3,3],numFilter[3],'conv4',activation_fn=leaky_relu)
	for n in range(nlayers):
		conv4 = layers.conv2d(conv4,[3,3],numFilter[3],'conv4-'+str(n),activation_fn=leaky_relu)
		
	pool4 = layers.max_pooling(conv4,[2,2],'pool4')
	
	conv5 = layers.conv2d(pool4,[3,3],numFilter[4],'conv5',activation_fn=leaky_relu)
	for n in range(nlayers):
		conv5 = layers.conv2d(conv5,[3,3],numFilter[4],'conv5-'+str(n),activation_fn=leaky_relu)
		
	pool5 = layers.max_pooling(conv5,[2,2],'pool5')
	
	up_conv6 = layers.deconv_upsample(pool5,2,'upsample6',activation_fn=leaky_relu)
	up_conv6 = layers.concat(up_conv6,conv5,'cont6')
	up_conv6 = layers.conv2d(up_conv6,[3,3],numFilter[3],'conv6',activation_fn=leaky_relu)
	for n in range(nlayers):
		up_conv6 = layers.conv2d(up_conv6,[3,3],numFilter[3],'conv6-'+str(n),activation_fn=leaky_relu)
	
	if shortCut == True:
		up_conv6f = layers.deconv_upsample(up_conv6,16,'upsample6f',activation_fn=leaky_relu)
		side6 = layers.conv2d(up_conv6f,[3,3],num_filters_in,'conv_side6',activation_fn=None)
		side6 = side6 + inputs
		side_outputs.append(side6)
	
	up_conv7 = layers.deconv_upsample(up_conv6,2,'upsample7',activation_fn=leaky_relu)
	up_conv7 = layers.concat(up_conv7,conv4,'cont7')
	up_conv7 = layers.conv2d(up_conv7,[3,3],numFilter[2],'conv7',activation_fn=leaky_relu)
	for n in range(nlayers):
		up_conv7 = layers.conv2d(up_conv7,[3,3],numFilter[2],'conv7-'+str(n),activation_fn=leaky_relu)
	
	if shortCut == True:
		up_conv7f = layers.deconv_upsample(up_conv7,8,'upsample7f',activation_fn=leaky_relu)	
		side7 = layers.conv2d(up_conv7f,[3,3],num_filters_in,'conv_side7',activation_fn=None)
		side7 = side7 + inputs
		side_outputs.append(side7)
	
	up_conv8 = layers.deconv_upsample(up_conv7,2,'upsample8',activation_fn=leaky_relu)
	up_conv8 = layers.concat(up_conv8,conv3,'cont8')
	up_conv8 = layers.conv2d(up_conv8,[3,3],numFilter[1],'conv8',activation_fn=leaky_relu)
	for n in range(nlayers):
		up_conv8 = layers.conv2d(up_conv8,[3,3],numFilter[1],'conv8-'+str(n),activation_fn=leaky_relu)
	
	if shortCut == True:
		up_conv8f = layers.deconv_upsample(up_conv8,4,'upsample8f',activation_fn=leaky_relu)
		side8 = layers.conv2d(up_conv8f,[3,3],num_filters_in,'conv_side8',activation_fn=None)
		side8 = side8 + inputs
		side_outputs.append(side8)
		
	up_conv9 = layers.deconv_upsample(up_conv8,2,'upsample9',activation_fn=leaky_relu)
	up_conv9 = layers.concat(up_conv9,conv2,'cont9')
	up_conv9 = layers.conv2d(up_conv9,[3,3],numFilter[0],'conv9',activation_fn=leaky_relu)
	for n in range(nlayers):
		up_conv9 = layers.conv2d(up_conv9,[3,3],numFilter[0],'conv9-'+str(n),activation_fn=leaky_relu)
	
	if shortCut == True:
		up_conv9f = layers.deconv_upsample(up_conv9,2,'upsample9f',activation_fn=leaky_relu)
		side9 = layers.conv2d(up_conv9f,[3,3],num_filters_in,'conv_side9',activation_fn=None)
		side9 = side9 + inputs
		side_outputs.append(side9)
	
	up_conv10 = layers.deconv_upsample(up_conv9,2,'upsample10',activation_fn=leaky_relu)
	up_conv10 = layers.concat(up_conv10,conv1,'cont10')
	
	up_conv10 = layers.conv2d(up_conv10,[3,3],num_filters_in,'conv10',activation_fn=None)
	for n in range(nlayers):
		up_conv10 = layers.conv2d(up_conv10,[3,3],num_filters_in,'conv10'+str(n),activation_fn=None)
		
	#outlayer = layers.conv2d(up_conv10,[3,3],num_filters_in,'conv10',activation_fn=leaky_relu)
	
	side10 = up_conv10 + inputs
	side_outputs.append(side10)
	
	if shortCut == True:
		nSide = len(side_outputs)
		weights = tf.get_variable('side_weight',[nSide],tf.float32,tf.constant_initializer(1.0/nSide))
		
		for n in range(nSide):
			if n == 0:
				final_layer = weights[n] * side_outputs[n]
			else:
				final_layer += weights[n] * side_outputs[n]
	else:
		final_layer = side10
	
	return side_outputs,final_layer
	
	
	
	
