import numpy as np
import os, cv2
from matplotlib import pyplot as plt
from CNNLorenzMie_Old.experiments.vmedian import vmedian

'''
pipeline for converting videos of experimental data to normalized images that are ready to feed into the models.
for each dataset, you should have a measurement video, a background video, and a darkcount video
function normalize_image returns list of normalized frames (in addition to saving them)
normalized images will be saved as 3-channel .png with the naming scheme:
norm_images/image0000.png
norm_images/image0001.png
in order of frames
'''



def normalize_video(bg_path, vid_path, dc_path, save_folder = './norm_images/', order = 2):
    #get first frame of background
    vidObj = cv2.VideoCapture(bg_path)
    success, img0 = vidObj.read()
    
    if not success:
        print('background video not found')
        return
    img0 = img0[:,:,0]

    print('Opening and computing background')
    #instantiate vmedian object
    v = vmedian(order=order, dimensions=img0.shape)
    v.add(img0)
    while success:
        success, image = vidObj.read()
        if success:
            image = image[:,:,0]
            v.add(image)
    #get background once video is done
    bg = v.get()

    #get first frame of darkcount
    dcObj = cv2.VideoCapture(dc_path)
    success, img0 = dcObj.read()
    img0 = img0[:,:,0]
    if not success:
        print('Dark count video not found')
        return

    print('Opening and computing darkcount')
    #instantiate vmedian object
    vdc = vmedian(order=order, dimensions=img0.shape)
    vdc.add(img0)
    while success:
        success, image = dcObj.read()
        if success:
            image = image[:,:,0]
            vdc.add(image)
    #get darkcount once video is done
    dc = vdc.get()

    print('Opening measurement video')

    #make save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    
    #get videocap object for measurement video
    vidObj = cv2.VideoCapture(vid_path)
    nframes = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    print(nframes, 'frames')
 
    print('Normalizing')
    #load and normalize measurement video
 #   img_return = []
    success = 1
    count=0
    vidObj.set(cv2.CAP_PROP_POS_FRAMES, count)
    frame = vidObj.get(cv2.CAP_PROP_POS_FRAMES)
    while success:
        success, image = vidObj.read()
        if success:
            numer =image[:,:,0] - dc
            denom = np.clip((bg-dc),1,255)
            testimg = np.divide(numer, denom)*100.
            testimg = np.clip(testimg, 0, 255)
            filename = os.path.dirname(save_folder) + '/image' + str(count).zfill(4) + '.png'
            cv2.imwrite(filename, testimg)
            testimg = np.stack((testimg,)*3, axis=-1)
#            img_return.append(testimg)

           # print(filename, end='\r')
            count+= 1
            print('Completed frame {}'.format(count), end='\r')

if __name__ == '__main__':
    dir = os.getcwd()
    bkgpath = dir+'/bg.avi'
    vidpath = dir+'/vid.avi'
    dcpath = dir +'/dc.avi'
    normalize_video(bkgpath, vidpath, dcpath)
