import os
import cv2
import matplotlib
import numpy as np

DIRECTORY_TRUTH =  '/home/srk/NTU/RealLifeDeceptionDetection.2016/Real-life_Deception_Detection_2016/Clips/Deceptive'
DIRECTORY_FALSE = '/home/srk/NTU/RealLifeDeceptionDetection.2016/Real-life_Deception_Detection_2016/Clips/Truthful'

def save_subjects(directory):

	dir_curr = directory.split('/')[-1]
	if not os.path.isdir(dir_curr):
		os.mkdir(dir_curr)

	for f in os.listdir(directory):
		#print(f)
		if f.endswith(".mp4"):
			cap=cv2.VideoCapture(directory + '/' + f)
			frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			fps = int(cap.get(cv2.CAP_PROP_FPS))

			num_seconds = frame_count // fps
			sampled_frame = int(num_seconds * fps / 2)
			cap.set(cv2.CAP_PROP_POS_FRAMES,sampled_frame)
			ret, frame = cap.read()

			cv2.imwrite(dir_curr + '/' + f.strip("mp4")+'png', frame)

if __name__=='__main__':
	save_subjects(DIRECTORY_FALSE)
	save_subjects(DIRECTORY_TRUTH)