from keras.models import Sequential, Model
from keras.layers import Conv2D, Activation, Dropout, Dense, Input, Reshape, Flatten, Lambda
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.utils.generic_utils import get_custom_objects
from keras.utils import to_categorical
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import keras as K
import keras.backend as kb
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

#from frames1 import filter_frames
from frames_alternate import get_matrices
from scipy import misc
from glob import glob
import csv


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
		

csvs_tmp = glob('data/CrowdDataset/*.csv')
csvs = []
for cs in csvs_tmp:
	if '06modified.csv' in cs:
		csvs.append(cs)
		
print(csvs)

"""	
data_sets = {}
for c in csvs:
	c.replace('\\','/')
	data_sets[c] = {}
	
	for h in range(24):
		with open(c,'r') as data:
			this_text = csv.reader(data,delimiter=';')
			for row in this_text:
				### Check if this area has been added
				if row[1] not in data_sets[c].keys():
					data_sets[c][row[1]] = {}
				## calculate second and hour of day
				times = row[0].split(':')[-1]
				second = int(float(times[-1]) / 100) + 60 * int(times[-2]) + 3600 * int(times[-3])
				hour = int(second/3600)
				
				if hour != h:
					continue
					
				### Check if array has been made for this hour 
				if int(second/3600) not in data_sets[c][row[1]]:
					data_sets[c][row[1]][hour] = np.zeros([60*60,1493,600],dtype=bool)
			
				x = int(int(row[2]) / 67)
				y = int(int(row[3]) / 67)
				
				data_sets[c][row[1]][hour][second-3600*hour][x][y] = 1

print( csvs )
input_data
inputs = Input(shape=input_data.shape)

opt = Adam(lr=1e-3)
dopt = Adam(lr=1e-4)
"""

grid_x = 160
grid_y = 80

dim4 = [int(grid_x/2**3),int(grid_y/2**3)]
dim3 = [int(grid_x/2**2),int(grid_y/2**2)]
dim2 = [int(grid_x/2),int(grid_y/2)]
dim1 = [int(grid_x),int(grid_y)]

### Multi-Scale Generators
def in_top_k(x):
	return tf.nn.in_top_k(x[0],x[0],x[1])
	
get_custom_objects().update({'in_top_k': Activation(in_top_k)})



def upscale_image(X):
	this_shape = tf.shape(X)
	return tf.image.resize_images(X,(this_shape[1]*2,this_shape[2]*2))
	
def get_upscale_output_shape(input_shape):
	this_shape = list(input_shape)
	this_shape[1] *= 2
	this_shape[2] *= 2
	return this_shape
	
def downscale_image(X):
	this_shape = tf.shape(X)
	return tf.image.resize_images(X,(this_shape[1]/2,this_shape[2]/2))
	
def get_downscaled_output_shape(input_shape):
	this_shape = list(input_shape)
	this_shape[1] /= 2
	this_shape[2] /= 2
	return this_shape
	
def concatenate_inputs(X):
	return X

def get_concatenate_shape(input_shape):
	return input_shape
	
N=0
def N_highest(X):
	orig_shape = X.shape
	X = X.ravel()
	top_k(X,N)

g_input = Input(dim1+[1])
g_input_1 = g_input
g_input_2 = Lambda(lambda x: tf.image.resize_images(x,dim2))(g_input)
g_input_3 = Lambda(lambda x: tf.image.resize_images(x,dim3))(g_input)
g_input_4 = Lambda(lambda x: tf.image.resize_images(x,dim4))(g_input)

downscaler_output_2_1 = Model(g_input, g_input_2)
downscaler_output_3_1 = Model(g_input, g_input_3)
downscaler_output_4_1 = Model(g_input, g_input_4)

g_input = Input(dim1+[4])
g_input_1 = g_input
g_input_2 = Lambda(lambda x: tf.image.resize_images(x,dim2))(g_input)
g_input_3 = Lambda(lambda x: tf.image.resize_images(x,dim3))(g_input)
g_input_4 = Lambda(lambda x: tf.image.resize_images(x,dim4))(g_input)

downscaler_output_2 = Model(g_input, g_input_2)
downscaler_output_3 = Model(g_input, g_input_3)
downscaler_output_4 = Model(g_input, g_input_4)

# First smallest scale
#g_input_4 = Input((*dim4,4))
t = Conv2D(128,3,padding='same',activation='relu')(g_input_4)
t = Conv2D(256,3,padding='same',activation='relu')(t)
t = Conv2D(128,3,padding='same',activation='relu')(t)
t = Conv2D(1,3,padding='same',activation='relu')(t)
t = Flatten()(t)
t = Dense(dim4[0]*dim4[1],activation='softmax')(t)

g_output_4 = Reshape(dim4+[1])(Dense(dim4[0]*dim4[1],activation='softmax')(t))
#g4 = Model(g_input_4,g_output_4,name='Scale_4_Generator')

# Second smallest scale
#g_input_3 = Input((*dim3,5))
#full_g_input_3 = K.backend.concatenate([g_input_3,Lambda(upscale_image,output_shape=get_upscale_output_shape)(g_output_4)])
#print(g4.get_layer(index=-1))
#g_input_3_2 = Lambda(concatenate_inputs,output_shape=get_concatenate_shape)(g_input_3)
#full_g_input_3 = concatenate([g_input_3_2,Lambda(upscale_image,output_shape=get_upscale_output_shape)(g_output_4)], axis=-1)
full_g_input_3 = concatenate([g_input_3,Lambda(upscale_image,output_shape=get_upscale_output_shape)(g_output_4)], axis=-1 )
t = Conv2D(128,5,padding='same',activation='relu')(full_g_input_3)
t = Conv2D(256,3,padding='same',activation='relu')(t)
t = Conv2D(128,3,padding='same',activation='relu')(t)
t = Conv2D(1,5,padding='same',activation='relu')(t)
t = Flatten()(t)
g_output_3 = Reshape(dim3+[1])(Dense(dim3[0]*dim3[1],activation='softmax')(t))
#g3 = Model(g_input_3,g_output_3,name='Scale_3_Generator')


# thrid smallest scale
#g_input_2 = Input((*dim2,5))
#full_g_input_2 = K.backend.concatenate([g_input_2,Lambda(upscale_image,output_shape=get_upscale_output_shape)(g3)])
full_g_input_2 = concatenate([g_input_2,Lambda(upscale_image,output_shape=get_upscale_output_shape)(g_output_3)], axis=-1 )
t = Conv2D(128,5,padding='same',activation='relu')(full_g_input_2)
t = Conv2D(256,3,padding='same',activation='relu')(t)
t = Conv2D(512,3,padding='same',activation='relu')(t)
t = Conv2D(256,3,padding='same',activation='relu')(t)
t = Conv2D(128,3,padding='same',activation='relu')(t)
t = Conv2D(1,5,padding='same',activation='relu')(t)
t = Flatten()(t)
g_output_2 = Reshape(dim2+[1])(Dense(dim2[0]*dim2[1],activation='softmax')(t))
#g2 = Model(g_input_2,g_output_2,name='Scale_2_Generator')


# Full scale
#g_input_1 = Input((*dim1,5))
#full_g_input_1 = K.backend.concatenate([g_input_1,Lambda(upscale_image,output_shape=get_upscale_output_shape)(g2)])
full_g_input_1 = concatenate([g_input_1,Lambda(upscale_image,output_shape=get_upscale_output_shape)(g_output_2)], axis=-1 )
t = Conv2D(128,7,padding='same',activation='relu')(full_g_input_1)
t = Conv2D(256,5,padding='same',activation='relu')(t)
t = Conv2D(512,5,padding='same',activation='relu')(t)
t = Conv2D(256,5,padding='same',activation='relu')(t)
t = Conv2D(128,5,padding='same',activation='relu')(t)
t = Conv2D(1,7,padding='same',activation='relu')(t)
t = Flatten()(t)
g_output_1 = Reshape(dim1+[1])(Dense(dim1[0]*dim1[1],activation='softmax')(t))
#g1 = Model(g_input_1,g_output_1,name='Scale_1_Generator')

#gs = [g4,g3,g2,g1]
generator_outputs = [g_output_4,g_output_3,g_output_2,g_output_1]
G = Model( g_input, outputs = generator_outputs )



####################################################################################################################




### Multi-Scale Discriminators
d_input_4 = Input(dim4+[5])
t = Conv2D(64,3,padding='valid',activation='relu')(d_input_4)
t = Flatten()(t)
t = Dense(512)(t)
t = Dense(256)(t)
d_output_4 = Dense(1,activation='sigmoid')(t)
d4 = Model(d_input_4,d_output_4,name='Scale_4_Discriminator')


d_input_3 = Input(dim3+[5])
t = Conv2D(64,3,padding='same',activation='relu')(d_input_3)
t = Conv2D(128,3,padding='same',activation='relu')(t)
t = Conv2D(128,3,padding='same',activation='relu')(t)
t = Flatten()(t)
t = Dense(1024)(t)
t = Dense(512)(t)
d_output_3 = Dense(1,activation='sigmoid')(t)
d3 = Model(d_input_3,d_output_3,name='Scale_3_Discriminator')


d_input_2 = Input(dim2+[5])
t = Conv2D(128,5,padding='same',activation='relu')(d_input_2)
t = Conv2D(256,5,padding='same',activation='relu')(t)
t = Conv2D(256,5,padding='same',activation='relu')(t)
t = Flatten()(t)
t = Dense(1024)(t)
t = Dense(512)(t)
d_output_2 = Dense(1,activation='sigmoid')(t)
d2 = Model(d_input_2,d_output_2,name='Scale_2_Discriminator')


d_input_1 = Input(dim1+[5])
t = Conv2D(128,7,padding='same',activation='relu')(d_input_1)
t = Conv2D(256,7,padding='same',activation='relu')(t)
t = Conv2D(512,5,padding='same',activation='relu')(t)
t = Conv2D(128,5,padding='same',activation='relu')(t)
t = Flatten()(t)
t = Dense(1024)(t)
t = Dense(512)(t)
d_output_1 = Dense(1,activation='sigmoid')(t)
d1 = Model(d_input_1,d_output_1,name='Scale_1_Discriminator')


discriminator_inputs = [d_input_4,d_input_3,d_input_2,d_input_1]
discriminator_outputs = [d_output_4,d_output_3,d_output_2,d_output_1]
D = Model( inputs = discriminator_inputs, outputs = discriminator_outputs )
D.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3))


def splice_output(X):
	return X[:,:,:,:4]
	
def get_splice_output_shape(input_shape):
	shape = list(input_shape)
	shape[-1] = 4
	return tuple(shape)

	
#############################################################################################################


def CreateDiscriminatorInput(gen,input,true_frame=None):
	if type(true_frame) == type(None):
		gen_data = gen.predict(input)
		g1_data = np.concatenate((input,gen_data[-1]),axis=-1)
		g2_data = np.concatenate((downscaler_output_2.predict(input),gen_data[-2]),axis=-1)
		g3_data = np.concatenate((downscaler_output_3.predict(input),gen_data[-3]),axis=-1)
		g4_data = np.concatenate((downscaler_output_4.predict(input),gen_data[-4]),axis=-1)
	else:
		g1_data = np.concatenate((input,true_frame),axis=-1)
		g2_data = np.concatenate((downscaler_output_2.predict(input),downscaler_output_2_1.predict(true_frame)),axis=-1)
		g3_data = np.concatenate((downscaler_output_3.predict(input),downscaler_output_3_1.predict(true_frame)),axis=-1)
		g4_data = np.concatenate((downscaler_output_4.predict(input),downscaler_output_4_1.predict(true_frame)),axis=-1)
	
	return [g4_data,g3_data,g2_data,g1_data]
	
	
### Complete GAN Model

g_inputs_list = [g_input_4,g_input_3,g_input_2,g_input_1]
d_inputs_list = [d_input_4,d_input_3,d_input_2,d_input_1]


d_connnected_to_g = [
	d4(K.layers.concatenate([g_input_4, g_output_4]) ),
	d3(K.layers.concatenate([g_input_3, g_output_3]) ),
	d2(K.layers.concatenate([g_input_2, g_output_2]) ),
	d1(K.layers.concatenate([g_input_1, g_output_1]) )]
	

GAN = Model( g_input, outputs = d_connnected_to_g )
#print(GAN.predict(np.random.uniform(0,1,size=[3,160,80,4])))
random_input = np.random.uniform(0,1,size=[3,160,80,4])
print()
#print(D.predict(CreateDiscriminatorInput(G,random_input)))
GAN.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3))
#GAN.summary()
a = []
print('getting matrices')
for i in range(1000):
	if i % 20 == 0:
		print('getting matrix i')
	new_frames = get_matrices()
	list_frames = []
	keys = list(new_frames.keys())
	keys.sort()
	for i in keys:
		list_frames.append(new_frames[i])
		
	frames_array = np.asarray(list_frames)
	a.append(frames_array)
a = np.asarray(a)
a = np.swapaxes(a,1,3)
a = np.swapaxes(a,1,2)
print(a.shape)	
X = a[:,:,:,:4]
Y = a[:,:,:,4:]
print(X.shape)
print(Y.shape)


discriminator_losses = []
generator_losses = []

for e in range(1000): 
	print(e)
	g_frames = G.predict(X)
	
	d_frames = CreateDiscriminatorInput(G,X)
	d_true_frames = CreateDiscriminatorInput(G,X,true_frame=Y)
	d_train = []
	for i in range(4):
		d_train.append(np.concatenate((d_frames[i],d_true_frames[i])))
	
	n_points = d_train[0].shape[0]
	
	y = np.zeros([n_points],dtype=np.int32)
	y[0:int(n_points/2)] = 0
	y[int(n_points/2):] = 1
	print(y.shape)
	print(y)
	#y = to_categorical(y)
	print(y.shape)
	print(y)
	make_trainable(D,True)
	d_loss  = D.train_on_batch(d_train,[y,y,y,y])
	discriminator_losses.append(sum(d_loss))

	
	
	y2 = np.full([int(n_points/2)],1,dtype=np.int32)
	#y2 = to_categorical(y2)

	make_trainable(D,False)
	g_loss = GAN.train_on_batch(X, [y2,y2,y2,y2] )
	generator_losses.append(sum(g_loss))
	print(g_loss)
	if e%100==0:
		plt.plot(range(e+1),discriminator_losses,label='D loss')
		plt.plot(range(e+1),generator_losses,label='G loss')
		plt.legend()
		plt.savefig(str(e)+'png')
		plt.close()
GAN.save('GAN.h5')	
D.save('D.h5')
exit()


#############################################################################################################



"""
### Attach all Components
opt=Adam(lr=1e-3)

### compile and attach all generators to each other
g4.compile(loss='binary_crossentropy',optimizer=opt)
g4_p = g4( Input((*dim4,4)) )

g3.compile(loss='binary_crossentropy',optimizer=opt)
g3_p = g3( 
	tf.concat((Input((*dim3,4)), tf.image.resize_images(g4_p,dim3)),
	axis=3) )
	
g2.compile(loss='binary_crossentropy',optimizer=opt)
g2_p = g2( 
	tf.concat((Input((*dim2,4)), tf.image.resize_images(g3_p,dim2)),
	axis=3) )

g1.compile(loss='binary_crossentropy',optimizer=opt)
g1_p = g1( tf.concat(
	(Input((*dim1,4)), tf.image.resize_images(g2_p,dim1)),
	axis=3) )



opt=Adam(lr=1e-3)

### compile discriminators and attach generators to discriminators
d4.compile(loss='binary_crossentropy',optimizer=opt)
d4_p = d4( g4_p )

d3.compile(loss='binary_crossentropy',optimizer=opt)
d3_p = d3( g3_p )

d2.compile(loss='binary_crossentropy',optimizer=opt)
d2_p = d2( g2_p )

d1.compile(loss='binary_crossentropy',optimizer=opt)









print(d1.summary())
#x1 = []
#for i in range(3000):
#	print(i)
#	x1.append( get_matrices() )

#print(len(x1))
#exit()
x = filter_frames()
print(x[64000])
t = [i for i in x.keys() if int(i) > 40000]
t.sort()
positions = [[ list(map(float,j[1:])) for j in x[i]] for i in t]
pid = [[j[0] for j in x[i]] for i in t]


position_maps= []
for i in range(len(t)):
	position_maps.append(np.zeros(dim1))
	for p in positions[i]:
		position_maps[-1][int(p[0])-850][int(p[1])-150] = 1




### Make sequences of 4 frams, and output of next frame
sequences = {1:[],2:[],3:[],4:[]}
next_frame = position_maps[4:]

### resizes batch of images (doesn't work for batch of sequences of images)
def res(images,scale):
	final = []
	for i in images:
		i = np.squeeze(i)
		final.append( misc.imresize(i,1.*2**scale) ) 
	return np.asarray(final)[:,:,:,np.newaxis]
	
### Downscales input for each scale
for i in range(len(position_maps)-4):
	seq = position_maps[i:i+4]
	sequences[4].append( list(map(lambda j: misc.imresize(j,.125), seq)) )
	sequences[3].append( list(map(lambda j: misc.imresize(j,.25), seq)) )
	sequences[2].append( list(map(lambda j: misc.imresize(j,.5), seq)) )
	sequences[1].append( position_maps[i:i+4] )

### convert sequences into tensor with appropriate dimensions
dims = {1:dim1,2:dim2,3:dim3,4:dim4}	
for i in (1,2,3,4):
	sequences[i] = np.asarray(sequences[i])
	sequences[i] = np.reshape(sequences[i],(len(sequences[i]),*dims[i],4))
#sequences = tf.convert_to_tensor(sequences,dtype=tf.float32)
#sequences = tf.expand_dims(sequences,axis=[-1])	
#print(sequences.get_shape())

sess = tf.Session()
x4 = sequences[4]
x3 = sequences[3]
x2 = sequences[2]
x1 = sequences[1]
xs = [x4,x3,x2,x1]





def probable_people(output,input):
	print(len(np.split(input,[1],axis=3)))
	print(len(np.split(input,[1],axis=3)[0]))
	print(len(np.split(input,[1],axis=3)[0][0]))
	print(len(np.split(input,[1],axis=3)[0][0][0]))
	print(len(np.split(input,[1],axis=3)[0][0][0][0]))

	N_people = np.sum(np.split(input,[1],axis=3)[0],axis=tuple(range(1,len(input.shape))))
	print(len(input[0]))
	print(input.shape)
	#print(N_people.shape)
	#probabilities = np.sort(output.ravel()).tolist()
	#print(probabilities)
	#print(len(probabilities))
	print(N_people[0])
	print(-1*N_people[1])
	probabilities = []
	c = 0
	for i in range(len(input)):
		#print()
		#print(c)
		#c+=1
		print( time.time() )
		#print('ravel')
		#a = output[i].ravel()
		#print(time.time())
		#print('sort')
		#np.sort(a).tolist()[-1*N_people[i]:]
		print(time.time())
		#print('altogether')
		probabilities.append(np.sort(output[i].ravel()).tolist()[-1*N_people[i]:])
		print(time.time())
		print()
		#print('probos')
		#print(probabilities[-1])
		#print(time.time())
	other = np.zeros(output.shape)
	for i in range(len(output)):
		print(time.time())
		for j in range(len(output[i])):
			for k in range(len(output[i][j])):
				if output[i][j][k][0] in probabilities[i]:
					other[i][j][k][0] = 1
					
	return other
	


print('calculating first noise bunch')
noise4 = g4.predict(x4,steps=1)
print('calculating second noise bunch')
noise_people4 = probable_people(noise4,x4[:len(noise4)])
print('actually doing second')
noise3 = g3.predict(np.concatenate((x3, res(noise4,1)),axis=3),steps=1)
print('calculating third noise bunch')
noise_people3 = probable_people(noise3,x4[:len(noise3)])
noise2 = g2.predict(np.concatenate((x2, res(noise3,1)),axis=3),steps=1)
print('calculating first noise bunch')
noise_people2 = probable_people(noise2,x4[:len(noise2)])
noise1 = g1.predict(np.concatenate((x1, res(noise2,1)),axis=3),steps=1)
noise_people1 = probable_people(noise1,x4[:len(noise1)])

noises = [noise4,noise3,noise2,noise1]

### train


ys = np.concatenate((np.zeros(len(xs[0])),np.full(len(xs[0]),1)))


for i in range(4):
	make_trainable(ds[i])
	ds.fit(np.concatenate((noises[i],xs[i]),axis=0),ys)
	make_trainable(ds[i])
	
	
"""





