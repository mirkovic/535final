import keras
import matplotlib.pyplot as plt 
from frames_alternate import get_matrices
import tensorflow as tf
model = keras.models.load_model('GAN.h5')
new_frames = get_matrices()
list_frames = []
keys = list(new_frames.keys())
keys.sort()
for i in keys:
	list_frames.append(new_frames[i])
	
frames_array = np.asarray(list_frames)
frames_array = np.swapaxes(frames_array,1,3)
frames_array = np.swapaxes(frames_array,1,2)
output = GAN.predict(frames_array[:,:,:,:4])

output_layer = GAN.get_layer(name='reshape_4')

sess = tf.Session()

output = sess.eval(output_layer)

exit()
plt.plot(frames_array[0,:,:,1])
plt.savefig('fram1.png')

plt.plot(frames_array[0,:,:,2])
plt.savefig('fram2.png')

plt.plot(frames_array[0,:,:,3])
plt.savefig('fram3.png')

plt.plot(frames_array[0,:,:,4])
plt.savefig('fram4.png')

plt.plot(frames_array[0,:,:,5])
plt.savefig('fram5.png')

plt.plot(output[0][0,:,:])
plt.savefig('fram5.png')