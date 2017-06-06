#!/usr/bin/env python
import json
import sys
import re
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as an
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from PIL import Image
import os
from glob import glob
import numpy as np
import pdb
from os.path import expanduser
import peakutils
import pedalProcessing

""" This script read in the pickle file and display images and make plots 
"""

sys.path.append(expanduser("~"))
from data_labeling_tool.data_sync import sync

def onClick(event):
    global pause
    pause ^= True
def sp_onClick(event):
	global delta
	delta += 2
	delta = max(delta,DELTA_MIN)
def sl_onClick(event):
	global delta
	delta -= 2
	#delta = min(delta,DELTA_MIN)
def r_label(label):
	global cid, radio1
	cid = label
	try:
		label_index = labels.index(label)
	except ValueError:
		pass
	[label.set_text(attr[label_index][num-1]) if (num > 0 and num < len(attr[label_index])+1) else label.set_text('NA') for num,label in enumerate(radio1.labels)]
	radio1.set_active(0)
	return cid
def r1_label(label):
	global clabel
	clabel = label
	return clabel
def log_label(event):
	print 'clabel: ', clabel, 'cid: ', cid
	if cid != 'NA' and clabel != 'NA':
		global data_label
		start_time = start.val
		stop_time = stop.val
		data_label.loc[pd.to_datetime(start_time, unit='us'):pd.to_datetime(stop_time, unit='us'),cid] = clabel
		print data_label.loc[pd.to_datetime(start_time, unit='us'):pd.to_datetime(stop_time, unit='us')]
		#dataframe = data_label[:pd.to_datetime(stop_time, unit='us')]
def save_label(event):
	print "Saving labels to file .. data_label.csv"
	dataframe = data_label.copy()
	dataframe.index = dataframe.index.astype(np.int64)//10**3
	dataframe.index.name = 'timestamp'
	dataframe.to_csv('data_label.csv')
	del dataframe	
def set_label_start(event):
	global start
	if int(drange.val) == 0:
		start.set_val(stop.val)
def press(event):
	global delta, press_delta
	#print('press', event.key)
	if event.key == 'right':
		delta = 0
		press_delta = 1
	if event.key == 'left':
		delta = 0
		press_delta = -1

def combine_imgs(img_files):
	imgs = [Image.open(img_file).rotate(180) if (type(img_file) in [str] and ind==rot_index)  else Image.open(img_file) if (type(img_file) in [str] and ind!=rot_index) else Image.open(img_files[valid_ind]) for ind, img_file in enumerate(list(img_files)) ]
	#min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs if np.sum(i.size)>0])[0][1]
	imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
	return Image.fromarray(imgs_comb)

if __name__ == "__main__":
	args = sys.argv[1:]
	file_name = args[0]
	if len(args) > 1: 
		labels = args[1:]
	else:
		labels = raw_input("Please enter your labels, space separated: ")
		labels = labels.split()
		
plot_labels = ['ssa', 'pbrk', 'hv_accp', 'sp1', 'b_p', 'yr', 'latitude', 'longitude', 'id', 'age', 'object_class', 'sizeY', 'throttle_position','leftlane_valid', 'leftlane_confidence', 'leftlane_boundarytype', 'rightlane_valid', 'rightlane_confidence','rightlane_boundarytype', 'face_cam', 'hand_cam', 'outside_cam']
img_labels = ['face_cam', 'hand_cam', 'outside_cam']

attr = []
attr_count = []
for num, label in enumerate(labels):
	attr.append(raw_input("Please enter attributes for label "+label+" (still space separated): ").split())
	attr_count.append(len(attr[num]))
num_class = max(attr_count)

file_name = os.path.abspath(file_name)
if os.path.isfile(file_name):
	print "Processing", os.path.split(file_name)[1]
	try:
		data = pd.read_csv(f)
		data['timestamp'] = pd.to_datetime(data['timestamp'], unit='us')
		data.set_index('timestamp', drop=True, inplace=True)
	except:
		with open(file_name, 'rb') as f:
			data = pickle.load(f)
	os.chdir(os.path.dirname(file_name))
elif os.path.isfile(file_name+'/'+os.path.split(file_name)[1]+'_extr_synced.csv'):
	print "Found an existing extracted csv file"
	print "Processing", os.path.split(file_name)[1]
	data = pd.read_csv(file_name+'/'+os.path.split(file_name)[1]+'_extr_synced.csv')
	data['timestamp'] = pd.to_numeric(data['timestamp'])#.convert_objects(convert_numeric=True)
	data['timestamp'] = pd.to_datetime(data['timestamp'], unit='us')
	data.set_index('timestamp', drop=True, inplace=True)
	os.chdir(file_name)
elif os.path.isfile(file_name+'/'+os.path.split(file_name)[1]+'_extr_synced.csv'):
	print "Found an existing extracted pickle file"
	print "Processing", os.path.split(file_name)[1]
	with open(file_name+'/'+os.path.split(file_name)[1]+'_extr_synced.pkl', 'rb') as f:
		data = pickle.load(f)
	os.chdir(file_name)
else:
	data = sync(file_name)
	os.chdir(file_name)
	
data = data.resample('100L').ffill()	
data.dropna(axis=0, how='all',inplace=True)

for item in plot_labels:
	if not item in data.columns:
		data[item] = 0

for item in img_labels:
	if not item in data.columns:
		img_labels.remove(item)
		
img_tstamp = [label+"_timestamp" for label in img_labels]
data_label = pd.DataFrame(columns=img_tstamp+labels, index=data.index)
data_label[img_tstamp] = data[img_labels].applymap(lambda x: str(x).split('/')[-1].split('.')[0])
#data_label = data_label.resample('100L').ffill()	
labels.insert(0, "NA")
attr.insert(0,["NA"]*(num_class+1))
pause = False
log = False
cid = labels[0]
clabel = attr[0][0]
delta = 0
press_delta = 0
DELTA_MIN = 0
start_time = data_label.index[0]
stop_time = data_label.index[0]
img_files = data.loc[data.index[0],img_labels]
rot_index = list(img_files.keys()).index('face_cam')
valid_imgs = [(ind, img) for ind, img in enumerate(list(img_files)) if str(img) != str(np.nan)]
valid_ind = valid_imgs[0][0]
min_shape = Image.open(valid_imgs[0][1]).size
plt_fig = plt.figure()
# Texts
ax = plt.subplot2grid((3,3), (1,2))
plt.setp(ax.get_xaxis(), visible=False)
plt.setp(ax.get_yaxis(), visible=False)
text = ax.text(0.05, 0.05, 'Detected object:', transform=ax.transAxes)
# Images
#ax1 = plt_fig.add_subplot(3,1,1)
ax1 = plt.subplot2grid((3,3), (0,0), colspan=2)
ax1.set_anchor('W')
#plt.setp(ax1.get_xticklabels(), visible=False)
#plt.setp(ax1.get_yticklabels(), visible=False)
plt.setp(ax1.get_xaxis(), visible=False)
plt.setp(ax1.get_yaxis(), visible=False)
plt_image = ax1.imshow(combine_imgs(img_files))
button = Button(ax1, '')
button.on_clicked(onClick)
# steering angle
#ax2 = plt_fig.add_subplot(3,3,4)
ax2 = plt.subplot2grid((3,3), (1,0))
plt_line, = ax2.plot([], [], 'r', label='Steering angle')
plt_line12, = ax2.plot([], [], 'r*', label='Left turn')
plt_line13, = ax2.plot([], [], 'bo', label='Right turn')
plt_line14, = ax2.plot([], [], 'k-', label='rightlane_curvature')
plt_line15, = ax2.plot([], [], 'k--', label='leftlane_curvature')
ax2.set_ylim(min(data['ssa'].min(), 0), data['ssa'].max())
ax2.set_xlim(0, data.shape[0]-1)
ax2.set_ylabel('Steering angle')
plt.legend(handles=[plt_line, plt_line12, plt_line13])
ssap = data['ssa'].copy()
ssan = data['ssa'].copy()
ssap[(ssap<90)] = 0
ssan[(ssan>-90)] = 0
ssap[data['sp1']<5] = 0
ssan[data['sp1']<5] = 0
ssap[ssap>=90] = ssap[ssap>=90]-90
ssan[ssan<=-90] = ssan[ssan<=-90]+90
#pdb.set_trace()
indexesl = peakutils.indexes(ssap.values.flatten(), thres=0, min_dist=300)
indexesr = peakutils.indexes(-ssan.values.flatten(), thres=0, min_dist=300)
tleft = np.array([200 if i in indexesl else np.nan for i in range(ssap.values.shape[0]) ])
tright = np.array([-200 if i in indexesr else np.nan for i in range(ssan.values.shape[0]) ])

print "Number of left turns: ", len(indexesl)
print "Number of right turns: ", len(indexesr)
# speed
ax3 = plt.subplot2grid((3,3), (1,1))
#plt_line1, = ax3.plot([], [], 'b', ms=1)
plt_line1, = ax3.plot([], [], 'b')
ax3.set_ylim(min(data['sp1'].min(), 0), data['sp1'].max())
ax3.set_xlim(0, data.shape[0]-1)
ax3.set_ylabel('Speed')
## Yaw rate
ax5 = plt.subplot2grid((3,3), (2,0))
#plt_line3, = ax5.plot([], [], 'k', label='Yaw Rate')
#plt_line6, = ax5.plot([], [], 'r', label='Brake Press')
plt_line8, = ax5.plot([], [], 'b', label='rightlane_offset')
plt_line9, = ax5.plot([], [], 'r', label='leftlane_offset')
plt_line10, = ax5.plot([], [], 'bo', label='right lane change')
plt_line11, = ax5.plot([], [], 'r*', label='left lane change')
#ax5.set_ylim(min(data[['yr', 'pbrk','rightlane_offset','leftlane_offset']].min().min(), 0), data[['yr', 'pbrk','rightlane_offset','leftlane_offset']].max().max())
ax5.set_ylim(min(data[['rightlane_offset','leftlane_offset']].min().min(), 0), data[['rightlane_offset','leftlane_offset']].max().max())
ax5.set_xlim(0, data.shape[0]-1)
ax5.set_ylabel('Left and Right Lane Offset')
#plt.legend(handles=[plt_line3, plt_line6, plt_line8, plt_line9])
plt.legend(handles=[plt_line8, plt_line9, plt_line10, plt_line11])
rightlane = data[['rightlane_offset']].copy().clip(lower=-3.0, upper=0.0).diff().values.flatten()
leftlane= data[['leftlane_offset']].copy().clip(lower=0.0, upper=3.0).diff().values.flatten()
leftch = np.nonzero(rightlane > 2.0)[0]+1
yleft = np.array([2 if i in leftch else np.nan for i in range(leftlane.shape[0]) ])
rightch = np.nonzero(leftlane < -2.0)[0]+1
yright = np.array([-2 if i in rightch else np.nan for i in range(leftlane.shape[0]) ])

print "Number of left lane changes: ", len(leftch)
print "Number of right lane changes: ", len(rightch)

## Throttle Position
ax6 = plt.subplot2grid((3,3), (2,1))
plt_line4, = ax6.plot([], [], 'r', label='Brake Pressure')
plt_line5, = ax6.plot([], [], 'b', label='Accel Pressure')
plt_line2, = ax6.plot([], [], 'r*', label= 'Braking')
plt_line6, = ax6.plot([], [], 'bo', label= 'Accel')
brake = data[['pbrk']].copy()
brake[brake>0.0001] = 1
brake[brake<=0.0001] = 0
brake = np.nonzero(brake.diff().values.flatten()>0)[0]+1
brake = [brake[i] for i in range(len(brake)-1) if brake[i+1]-brake[i] > 250]
accel = data[['hv_accp']].copy()
accel[accel>0.0001] = 1
accel[accel<=0.0001] = 0
accel = np.nonzero(accel.diff().values.flatten()>0)[0]+1
accel = [accel[i] for i in range(len(accel)-1) if accel[i+1]-accel[i] > 100]
ybrake = np.array([2 if i in brake else np.nan for i in range(data.shape[0]) ])
yaccel = np.array([2 if i in accel else np.nan for i in range(data.shape[0]) ])

print "Number of braking events: ", len(brake)
print "Number of accel events: ", len(accel)
"""ybrake, yaccel = pedalProcessing.processPedalData(data['pbrk'].values.tolist(), data['hv_accp'].values.tolist(),20,0.1,0)
ybrake = 2*ybrake
yaccel = 2*yaccel"""

ax6.set_ylim(0, data[['pbrk','hv_accp']].max().max()/10)
ax6.set_xlim(0, data.shape[0]-1)
ax6.set_ylabel('Brake And Accel. Pressure')
plt.legend(handles=[plt_line4, plt_line5, plt_line2, plt_line6])

pdb.set_trace()

## Log and lat
ax7 = plt.subplot2grid((3,3), (2,2))
plt_line7, = ax7.plot([], [], 'r')
ax7.set_ylim(data['latitude'].min(), data['latitude'].max())
ax7.set_xlim(data['longitude'].min(), data['longitude'].max())
ax7.set_ylabel('Latitude')
ax7.set_xlabel('Longitude')
ax8 = plt.axes([0.47+0.2, 0.86, 0.05, 0.04])
spbutton = Button(ax8, 'FFW >>', color='lightblue', hovercolor='0.975')
spbutton.on_clicked(sp_onClick) 
ax9 = plt.axes([0.47+0.2, 0.81, 0.05, 0.04])
slbutton = Button(ax9, 'RWD <<', color='magenta', hovercolor='0.975')
slbutton.on_clicked(sl_onClick)
ax10 = plt.axes([0.47+0.2, 0.76, 0.05, 0.04])
readybutton = Button(ax10, 'Set label \nstart time', color='yellow', hovercolor='0.975')
readybutton.on_clicked(set_label_start)
ax11 = plt.axes([0.47+0.2, 0.71, 0.05, 0.04])
logbutton = Button(ax11, 'Stop time \nLog label', color='green', hovercolor='0.975')
logbutton.on_clicked(log_label)
ax12 = plt.axes([0.87, 0.81, 0.05, 0.09])
savebutton = Button(ax12, 'Save labels\nto file', color='cyan', hovercolor='0.975')
savebutton.on_clicked(save_label)
startax = plt.axes([0.15, 0.625, 0.25, 0.03])
stopax = plt.axes([0.55, 0.625, 0.25, 0.03])
start = Slider(startax, 'Label Start \nTimestamp', data.index[0].value//10**3, data.index[-1].value//10**3, valinit=data.index[0].value//10**3)
stop = Slider(stopax, 'Label Stop\nTimestamp', data.index[0].value//10**3, data.index[-1].value//10**3, valinit=data.index[0].value//10**3)
start.valfmt = '%i'
stop.valfmt = '%i'
rangeax = plt.axes([0.47+0.2, 0.67, 0.18, 0.03])
drange = Slider(rangeax,'Constant\ntimedelta(s)', -100, 100, valinit=0)
drange.valfmt = '%i'
rax = plt.axes([0.53+0.2, 0.71, 0.06, 0.19])
radio = RadioButtons(rax, labels, active=0)
#radio = CheckButtons(rax, labels[1:], [False]*len(labels[1:]))
rax1 = plt.axes([0.60+0.2, 0.71, 0.06, 0.19])
radio1 = RadioButtons(rax1, ['NA']*(num_class+1), active=0)
#radio1 = CheckButtons(rax1, range(num_class), [False]*num_class)
radio.on_clicked(r_label)
radio1.on_clicked(r1_label)
def itergen():
	global press_delta
	i_max = data.shape[0]
	i = 0
	while i+delta < i_max:
		if not pause:
			i+=delta
		i+= press_delta
		press_delta = 0
		yield i
def updatefig(itergen):
	i = itergen
	stop.set_val(data_label.index[i].value//10**3)
	if int(drange.val) != 0:
		start.set_val(data_label.index[i].value//10**3-int(drange.val)*10**6)
	global img_files
	img_files = data.loc[data.index[i],img_labels]
	plt.suptitle('Processing timestamp '+str((data.index[i]))+', remaining '+str(data.index[-1]-data.index[i]).split()[-1]+'/'+str(data.index[-1]-data.index[0]).split()[-1]+' to complete trip', fontsize=14)
	plt_image.set_array(combine_imgs(img_files))
	plt_line.set_data(range(i), data.loc[data.index[:i], 'ssa'].values)
	plt_line1.set_data(range(i), data.loc[data.index[:i], 'sp1'].values)
	#plt_line3.set_data(range(i), data.loc[data.index[:i], 'yr'].values)
	plt_line4.set_data(range(i), data.loc[data.index[:i], 'pbrk'].values)
	plt_line5.set_data(range(i), data.loc[data.index[:i], 'hv_accp'].values/10)
	plt_line8.set_data(range(i), data.loc[data.index[:i], 'rightlane_offset'].values)
	plt_line9.set_data(range(i), data.loc[data.index[:i], 'leftlane_offset'].values)
	plt_line2.set_data(range(i), ybrake[:i])
	plt_line6.set_data(range(i), yaccel[:i])
	plt_line10.set_data(range(i), yright[:i])
	plt_line11.set_data(range(i), yleft[:i])
	plt_line12.set_data(range(i), tleft[:i])
	plt_line13.set_data(range(i), tright[:i])
	#plt_line14.set_data(range(i), 50000*data.loc[data.index[:i], 'rightlane_curvature'].values)
	#plt_line15.set_data(range(i), 50000*data.loc[data.index[:i], 'leftlane_curvature'].values)
	ax2.set_xlim(max(0,i-900), min(i+100, data.shape[0]-1))
	ax3.set_xlim(max(0,i-900), min(i+100, data.shape[0]-1))
	ax5.set_xlim(max(0,i-900), min(i+100, data.shape[0]-1))
	ax6.set_xlim(max(0,i-900), min(i+100, data.shape[0]-1))
	plt_line7.set_data(data.loc[data.index[:i], 'longitude'].values, data.loc[data.index[:i], 'latitude'].values)
	text.set_text('Detected object: '+str(data.loc[data.index[i], 'object_class'])+'\n'+
	'Object id: '+str(data.loc[data.index[i], 'id'])+'\n'+
	'Object size: '+str(data.loc[data.index[i], 'sizeY'])+'\n'+
	'Object age: '+str(data.loc[data.index[i], 'age'])+'\n'+
	'Leftlane_valid: '+str(data.loc[data.index[i], 'leftlane_valid'])+'\n'+
	'Rightlane_valid: '+str(data.loc[data.index[i], 'rightlane_valid'])+'\n'+
	'Leftlane_confidence: '+str(data.loc[data.index[i], 'leftlane_confidence'])+'\n'+
	'Rightlane_confidence: '+str(data.loc[data.index[i], 'rightlane_confidence'])+'\n'+
	'Leftlane_boundarytype: '+str(data.loc[data.index[i], 'leftlane_boundarytype'])+'\n'+
	'Rightlane_boundarytype: '+str(data.loc[data.index[i], 'rightlane_boundarytype'])+'\n'+
	'Replay Speed: '+str(delta)+'X\n'+
	'Label timedelta: '+str(pd.to_timedelta(stop.val-start.val, unit='us')).split()[-1]+'\n'
	) 
		
	return plt_image, plt_line, plt_line1, plt_line2, plt_line4, plt_line5, plt_line6, plt_line7, plt_line8, plt_line9,text, start, stop, data_label, ax2, ax3, ax5, ax6 
plt_fig.canvas.mpl_connect('key_press_event', press)
#plt_fig.canvas.mpl_connect('button_press_event', onClick)
plt_ani = an.FuncAnimation(plt_fig, updatefig, itergen, interval = 10, blit=False, repeat=False)
plt.show()
