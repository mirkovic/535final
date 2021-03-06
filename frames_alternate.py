from glob import glob
from collections import defaultdict
import os, time, random, numpy as np, csv as CSV
import matplotlib.pyplot as plt

raw_frames = defaultdict()
t_window, room = 1, 'PIW' # length of time captured by a frame, in seconds
csv = [s.replace('\\', '/') for s in glob('./data/*02-10.csv')][0]

class Window(object):
	def __init__(self, o_x, o_y, w, h):
		self.origin = (o_x, o_y)
		ext = setup_room()
		self.x_lo, self.x_hi = o_x, o_x + w
		self.y_lo, self.y_hi = ext[3] - o_y, ext[3] - (o_y + h) # y_lo > y_hi
	def in_bounds(self, point):
		x, y = point
		return float(self.x_lo) < float(x) < float(self.x_hi) \
			and float(self.y_hi) < float(y) < float(self.y_lo)

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

def get_frames(RW, name=csv[:-4], valid_room=room):
	frames, people = defaultdict(list), defaultdict()
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
			for p_id, frame_list in people.items():
				for frame_id, positions in frame_list.items():
					# record one person's avg location within a frame
					avg_loc = np.array(positions).mean(axis=0)
					frames[frame_id].append((p_id, avg_loc))
					f.write('{},{},{},{}\n'.format(frame_id, p_id, avg_loc[0], avg_loc[1]))
		
	if RW == 'r': # read existing csv
		with open('{}-frames-{}s.csv'.format(name, t_window), 'r') as c:
			frame_list = CSV.reader(c, delimiter=',')
			for frame_id, p_id, avg_x, avg_y in frame_list:
				frames[frame_id].append((p_id, (float(avg_x), float(avg_y))))

	return frames

# Show the two clusters of people per frame of time
def display_frames(frames):
	plt.scatter([k for k in frames.keys() if int(k) < 30000], 
		[len(v) for k, v in frames.items() if int(k) < 30000])
	plt.show()
	plt.scatter([k for k in frames.keys() if int(k) > 50000], 
		[len(v) for k, v in frames.items() if int(k) > 50000])
	plt.show()

# Plot image
def setup_room(valid_room=room):
	ext = get_extent(raw_frames)
	img = plt.imread("./data/{}.jpg".format(valid_room))
	aspect = img.shape[0]/float(img.shape[1])*((ext[1]-ext[0])/(ext[3]-ext[2]))
	plt.imshow(img, zorder=0, extent=ext)
	plt.gca().set_aspect(aspect)
	return ext

# Plot frame on a room
def display_room(frames, frame_id, at_once=True, n_future=5):
	ext = setup_room()
	plt.title('Frame {}'.format(frame_id))
	print('frame {} has {} people in it'.format(frame_id, len(frames[str(frame_id)])))
	for f_id in range(int(frame_id), int(frame_id) + n_future):
		for p_id, avg_loc in frames[str(f_id)]:
			print('\tperson: {}\tloc: {}'.format(p_id, avg_loc))
			plt.scatter(float(avg_loc[0]), ext[3]-float(avg_loc[1]), zorder=1, s=10)
		if not at_once: plt.show(), setup_room()
	if at_once: plt.show()

# Show bounds on room by grid
def show_bounds(grid):
	setup_room()
	plt.scatter(grid.x_lo, grid.y_lo)
	plt.scatter(grid.x_lo, grid.y_hi)
	plt.scatter(grid.x_hi, grid.y_lo)
	plt.scatter(grid.x_hi, grid.y_hi)
	plt.show()

# Filter frames to show locations only within a window
def filter_frames(frames, grid):
	new_frames = defaultdict(list)
	for frame_id, avg_locs in frames.items():
		for p_id, avg_loc in avg_locs:
			if grid.in_bounds(avg_loc):
				new_frames[frame_id].append((p_id, avg_loc))
	return new_frames

# Choose a frame for which n_people is constant in the next n_future frames
def select_frame(frames, n_future, min_ppl):
	const = False
	while not const:
		frame_id = random.choice([x for x in frames.keys() if int(x) > 40000])
		if len(frames[frame_id]) < min_ppl: continue
		n_people = len(frames[frame_id])
		const = True
		for f_id in range(int(frame_id), int(frame_id) + n_future):
			if not str(f_id) in frames: const = False; break
			if len(frames[str(f_id)]) != n_people: const = False; break
		if const: return frame_id
	return -1 # none exist

# Get dict of frames in a range [f_lo, f_hi)
def get_frame_range(frames, f_lo, n_future):
	return {str(f_id):frames[str(f_id)] for f_id in range(int(f_lo), int(f_lo) + n_future)}

# Get matrix corresponding to a single frame
def frame_to_matrix(frames, frame_id, grid):
	A = np.zeros((160, 80))
	for p_id, avg_loc in frames[frame_id]:
		x, y = round(float(avg_loc[0])), round(avg_loc[1])
		A[int(x-grid.origin[0]-1)][int(y-grid.origin[1]-1)] = 1
	return A

# get matrix reps of a set of n_future frames with const # ppl > min_ppl
def get_matrices(origin=(850, 150), dimensions=(160, 80), n_future=5, min_ppl=5,display=False):
	global raw_frames
	if len(raw_frames) == 0: raw_frames = get_frames(RW='r')
	grid = Window(o_x=origin[0], o_y=origin[1], w=dimensions[0], h=dimensions[1])
	frames = filter_frames(frames=raw_frames, grid=grid)
	frame_id = select_frame(frames=frames, n_future=n_future, min_ppl=min_ppl)
	chosen = get_frame_range(frames=frames, f_lo=frame_id, n_future=n_future)
	if display: display_room(chosen, frame_id, at_once=False)
	return {f_id:frame_to_matrix(frames=frames, frame_id=f_id, grid=grid) for f_id in chosen.keys()}

if __name__ == '__main__':
	n_future, min_ppl = 5, 5
	origin, dimensions = (850, 150), (160, 80)
	m = get_matrices(origin, dimensions, n_future, min_ppl, display=True)
	#frames = get_frames(RW='r')
	#grid = Window(o_x=850, o_y=150, w=160, h=80)
	#new_frames = filter_frames(frames=frames, grid=grid)
	#frame_id = select_frame(frames=new_frames, n_future=5)
	#chosen_frames = get_frame_range(frames=new_frames, f_lo=frame_id, n_future=5)
	#matrices = {f_id:frame_to_matrix(frames=new_frames, frame_id=f_id, grid=grid) for f_id in chosen_frames.keys()}
	#show_bounds(grid=grid)
	#display_room(frame_id=frame_id, frames=new_frames, at_once=False)
	#display_frames(frames)
	#print('frame_id: {}\tn_people: {}'.format(frame_id, len(new_frames[frame_id])))
	