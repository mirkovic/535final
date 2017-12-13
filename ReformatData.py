from glob import glob
import csv
import os
import time

csvs_tmp = glob('data/CrowdDataset/*.csv')
csvs = []
for c in csvs_tmp:
	if 'modified' not in c and '19' in c:
		csvs.append(c)
data_sets = {}

for c in csvs:
	print(c)
	
	c.replace('\\','/')
	
	try:
		os.remove(c[:-4]+'modified.csv')
	except OSError:
		pass
	
	
	#for h in range(24):
	for h in range(24):
		data_sets = {}
		with open(c,'r') as data:
			this_text = csv.reader(data,delimiter=';')
			print('hey')
			num = 0
			for row in this_text:
				num+=1
				### Check if this area has been added
				try:
					if row[1] not in data_sets.keys():
						data_sets[row[1]] = {}
				
					## calculate second and hour of day
					times = row[0].split(':')
				
					second = int(round(float(times[-1]) / 100)) + 6 * int(times[-2]) + 360 * int(times[-3])
				
					hour = int(second/3600)
					if hour != h:
						continue
				
					if second not in data_sets[row[1]]:
						data_sets[row[1]][second] = row[1]+','+str(second) + ','
			
					x = int(int(row[2]) / 67)
					y = int(int(row[3]) / 67)
					data_sets[row[1]][second] += str(x)+','+str(y)+','
					if num % 1000000 == 0:
						time.sleep(1)
				except:
					print(c)
					print( 'line reading error. line '+str(num)+':',row)
					continue
		with open(c[:-4]+'modified.csv','a') as thing:
			for area in data_sets:
				for sec in sorted(data_sets[area].keys()):
					thing.write(data_sets[area][sec][:-1]+'\n')
					

				