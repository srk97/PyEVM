import argparse

import cv2
import numpy as np
import scipy.fftpack as fftpack
import scipy.signal as signal

from ..hparams.registry import get_hparams

parser = argparse.ArgumentParser()
parser.add_argument("hparams", type=str)
args = parser.parse_args()
#global hps
hps = get_hparams(args.hparams)

#convert RBG to YIQ
def rgb2ntsc(src):
    [rows,cols]=src.shape[:2]
    dst=np.zeros((rows,cols,3),dtype=np.float64)
    T = np.array([[0.114, 0.587, 0.298], [-0.321, -0.275, 0.596], [0.311, -0.528, 0.212]])
    for i in range(rows):
        for j in range(cols):
            dst[i, j]=np.dot(T,src[i,j])
    return dst

#convert YIQ to RBG
def ntsc2rbg(src):
    [rows, cols] = src.shape[:2]
    dst=np.zeros((rows,cols,3),dtype=np.float64)
    T = np.array([[1, -1.108, 1.705], [1, -0.272, -0.647], [1, 0.956, 0.620]])
    for i in range(rows):
        for j in range(cols):
            dst[i, j]=np.dot(T,src[i,j])
    return dst

#Build Gaussian Pyramid
def build_gaussian_pyramid(src,level=3):
    s=src.copy()
    pyramid=[s]
    for _ in range(level):
        s=cv2.pyrDown(s)
        pyramid.append(s)
    return pyramid

#Build Laplacian Pyramid
def build_laplacian_pyramid(src,levels=3):
    gaussianPyramid = build_gaussian_pyramid(src, levels)
    pyramid=[]
    for i in range(levels,0,-1):
        GE=cv2.pyrUp(gaussianPyramid[i])
        L=cv2.subtract(gaussianPyramid[i-1],GE)
        pyramid.append(L)
    return pyramid

#load video from file
def load_video(video_filename, levels):
    cap=cv2.VideoCapture(video_filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    resized_ht = height - (height%(2**levels))
    resized_wt = width - (width%(2**levels))
    video_tensor=np.zeros((frame_count,resized_ht,resized_wt,3),dtype='float')
    x=0
    while cap.isOpened():
        ret,frame=cap.read()
        if ret is True:
            video_tensor[x] = cv2.resize(frame, (resized_wt,resized_ht))
            x+=1
        else:
            break
    return video_tensor,fps

# apply temporal ideal bandpass filter to gaussian video
def temporal_ideal_filter(tensor,low,high,fps,axis=0):
    fft=fftpack.fft(tensor,axis=axis)
    frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff=fftpack.ifft(fft, axis=axis)
    return np.abs(iff)

# build gaussian pyramid for video
def gaussian_video(video_tensor,levels=3):
    for i in range(0,video_tensor.shape[0]):
        frame=video_tensor[i]
        pyr=build_gaussian_pyramid(frame,level=levels)
        gaussian_frame=pyr[-1]
        if i==0:
            vid_data=np.zeros((video_tensor.shape[0],gaussian_frame.shape[0],gaussian_frame.shape[1],3))
        vid_data[i]=gaussian_frame
    return vid_data

#amplify the video
def amplify_video(gaussian_vid,amplification=50):
    return gaussian_vid*amplification

#reconstruct video from original video and gaussian video
def reconstruct_video(amp_video,origin_video,levels=3):
    final_video=np.zeros(origin_video.shape)
    for i in range(0,amp_video.shape[0]):
        img = amp_video[i]
        for _ in range(levels):
            img=cv2.pyrUp(img)
        img=img+origin_video[i]
        final_video[i]=img
    return final_video

#save video to files
def save_video(video_tensor, saved_name, modew='motion'):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    [height,width]=video_tensor[0].shape[0:2]
    writer = cv2.VideoWriter(saved_name + '_' + modew+'.avi', fourcc, 30, (width, height), 1)
    for i in range(0,video_tensor.shape[0]):
        writer.write(cv2.convertScaleAbs(video_tensor[i]))
    writer.release()

#magnify color
def magnify_color(video_name, hps, saved_name=None):
    if saved_name is None:
        saved_name = video_name.strip('mp4')[:-1]
    t,f=load_video(video_name, hps.levels)
    #print(t.shape, f)
    gau_video=gaussian_video(t,levels=hps.levels)
    filtered_tensor=temporal_ideal_filter(gau_video,hps.cutoff_low,hps.cutoff_high,f)
    amplified_video=amplify_video(filtered_tensor,amplification=hps.amplification)
    final=reconstruct_video(amplified_video,t,levels=hps.levels)
    save_video(final, saved_name, modew='color')

#build laplacian pyramid for video
def laplacian_video(video_tensor,levels=3):
    tensor_list=[]
    for i in range(0,video_tensor.shape[0]):
        frame=video_tensor[i]
        pyr=build_laplacian_pyramid(frame,levels=levels)
        if i==0:
            for k in range(levels):
                tensor_list.append(np.zeros((video_tensor.shape[0],pyr[k].shape[0],pyr[k].shape[1],3)))
        for n in range(levels):
            tensor_list[n][i] = pyr[n]
    return tensor_list

#butterworth bandpass filter
#Reference: https://stackoverflow.com/a/20935536
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    omega = 0.5 * fs
    low = lowcut / omega
    high = highcut / omega
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data, axis=0)
    return y

#reconstruct video from laplacian pyramid
def reconstruct_from_tensorlist(filter_tensor_list,levels=3):
    final=np.zeros(filter_tensor_list[-1].shape)
    for i in range(filter_tensor_list[0].shape[0]):
        up = filter_tensor_list[0][i]
        for n in range(levels-1):
            up=cv2.pyrUp(up)+filter_tensor_list[n + 1][i]
        final[i]=up
    return final

#manify motion
def magnify_motion(video_name, hps, saved_name=None):
    print(hps.levels)
    print(video_name)
    if saved_name is None:
        saved_name = video_name.strip('mp4')[:-1]
    t,f=load_video(video_name, hps.levels)
    #print(t.shape, f)
    lap_video_list=laplacian_video(t,levels=hps.levels)
    filter_tensor_list=[]
    for i in range(hps.levels):
        filter_tensor=butter_bandpass_filter(lap_video_list[i],hps.cutoff_low,hps.cutoff_high,f)
        filter_tensor*=hps.amplification
        filter_tensor_list.append(filter_tensor)
    recon=reconstruct_from_tensorlist(filter_tensor_list)
    final=t+recon
    save_video(final, saved_name)

if __name__=="__main__":

    for fn in hps.mode:
        locals()[fn]("/home/srk/NTU/PyEVM/misc/test_videos/trial_lie_038.mp4", hps)
