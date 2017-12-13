from keras.models import Sequential, Model
from keras.layers import Conv2D, Activation, Dropout, Dense, Input
from keras.optimizers import Adam

import tensorflow as tf
import numpy as np

from scipy import misc
from glob import glob
import csv


csvs_tmp = glob('data/CrowdDataset/*.csv')
csvs = []
for cs in csvs_tmp:
	if '06modified.csv' in cs:
		csvs.append(cs)
		
print(csvs)
exit()
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
				print(x)
				print(y)
				print(hour)
				print(second)
				data_sets[c][row[1]][hour][second-3600*hour][x][y] = 1

print( csvs )
input_data
inputs = Input(shape=input_data.shape)

opt = Adam(lr=1e-3)
dopt = Adam(lr=1e-4)

#Generator
Dense(16, activation='relu')(inputs)
Dense(32, activation='relu')()




d_input = Input(shape=shp)
H = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(d_input)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Convolution2D(512, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Flatten()(H)
H = Dense(256)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
d_V = Dense(2,activation='softmax')(H)
