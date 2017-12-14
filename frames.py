from glob import glob
from collections import defaultdict
import os, time, random, numpy as np, csv as CSV
import matplotlib.pyplot as plt

t_window = 1 # length of time captured by a frame, in seconds
csv = [s.replace('\\', '/') for s in glob('./data/*02-10.csv')][0]
frames = defaultdict(list) # one per hour
people = defaultdict()
valid_room = 'PIW'
name = csv[:-4]
peep_lim = None
verbose = True
RW = 'r' #'w'

# input: ['2013-02-10T07:18:53:050', 'PIW', '68064', '11071', '263']
def row_to_dict(row): # output: tracklet dict
	date_time = row[0].split('T') # ['2013-02-10','07:18:53:050']
	hr, mn, sec, ms = date_time[-1].split(':') # ['07','18','53','050']
	return {'room': row[1], 'id': int(row[-1]),
			'pos': np.array([int(row[-3]), int(row[-2])]),
			'date': date_time[0], 'hr': hr, 'min': mn, 'sec': sec, 'ms': ms}

# assumes unique pairs of hr, min, sec
def frame_id(tracklet):
	uniq_sec = int(tracklet['hr'])*3600000 \
			 + int(tracklet['min'])*60000 \
			 + int(tracklet['sec'])*1000 \
			 + int(tracklet['ms'])
	return int(uniq_sec/(t_window*1000))

# get the boundaries of all frames in the dict
def get_extent(frames):
	x_min, x_max, y_min, y_max = 2**20, -2**20, 2**20, -2**20
	for frame_id, avg_locations in frames.items():
		for p_id, avg_loc in avg_locations:
			if float(avg_loc[0]) < x_min: x_min = float(avg_loc[0])
			if float(avg_loc[0]) > x_max: x_max = float(avg_loc[0])
			if float(avg_loc[1]) < y_min: y_min = float(avg_loc[1])
			if float(avg_loc[1]) > y_max: y_max = float(avg_loc[1])
	return [x_min, x_max, y_min, y_max]

if RW == 'w': # create csv from parsed data
	with open('{}-frames-{}s.csv'.format(name, t_window), 'a') as f:
		with open(csv, 'r') as c:
			body = CSV.reader(c, delimiter=';')
			for tracklet in body: # tracklets
				t = row_to_dict(tracklet)
				if t['room'] != valid_room: continue
				if not t['id'] in people:
					people[t['id']] = defaultdict(list) # {sec: [locs]}
				people[t['id']][frame_id(t)].append(t['pos']/67)
				if verbose: print(t)
				if peep_lim and t['id'] % peep_lim == 0: break # testing
		for p_id, frame_list in people.items():
			for frame_id, positions in frame_list.items():
				# record one person's avg location within a frame
				avg_loc = np.array(positions).mean(axis=0)
				frames[frame_id].append((p_id, avg_loc))
				if not peep_lim: f.write('{},{},{},{}\n'.format(frame_id, p_id, avg_loc[0], avg_loc[1]))
	
if RW == 'r': # read existing csv
	with open('{}-frames-{}s.csv'.format(name, t_window), 'r') as c:
		frame_list = CSV.reader(c, delimiter=',')
		for frame_id, p_id, avg_x, avg_y in frame_list:
			frames[frame_id].append((p_id, (avg_x, avg_y)))

# Show the two clusters of people per frame of time
def display_frames():
	plt.scatter([k for k in frames.keys() if int(k) < 30000], [len(v) for k, v in frames.items() if int(k) < 30000])
	plt.show()
	plt.scatter([k for k in frames.keys() if int(k) > 50000], [len(v) for k, v in frames.items() if int(k) > 50000])
	plt.show()

# Plot image
def setup_room():
	ext = get_extent(frames)
	img = plt.imread("./data/{}.jpg".format(valid_room))
	aspect = img.shape[0]/float(img.shape[1])*((ext[1]-ext[0])/(ext[3]-ext[2]))
	plt.imshow(img, zorder=0, extent=ext)
	plt.gca().set_aspect(aspect)
	return ext

# Plot frame on a room
def display_room(frame_id, at_once=True):
	ext = setup_room()
	plt.title('Frame {}'.format(frame_id))
	print('frame {} has {} people in it'.format(frame_id, len(frames[str(frame_id)])))
	for f_id in range(int(frame_id), int(frame_id)+5):
		for p_id, avg_loc in frames[str(f_id)]:
			print('\t{}\t{}'.format(p_id, avg_loc))
			plt.scatter(float(avg_loc[0]), ext[3]-float(avg_loc[1]), zorder=1, s=10)
		if not at_once: plt.show(), setup_room()
	if at_once: plt.show()


# 1. Overlay average locations onto an image/room to generate a view of a frame
frame_id = random.choice([x for x in frames.keys() if int(x) > 40000])
display_room(frame_id, at_once=False)

# 2. Obtain all frames pertaining to a specific 320 x 320 pixel view of the image/room.

