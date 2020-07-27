import cv2
import numpy as np
import glob

#https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
def video2frames():
    #vidcap = cv2.VideoCapture('E:\\inf_master\\Semester-5\\Komplexpraktikum\\code\\video\\Course3_Trial_3_109_Stitching.avi')
    #vidcap = cv2.VideoCapture('E:\\inf_master\\Semester-5\\Komplexpraktikum\\code\\video\\Course1_Trial_3_11_Gallbladder.avi')
    #vidcap = cv2.VideoCapture('E:\\inf_master\\Semester-5\\Komplexpraktikum\\code\\video\\Course1_Trial_3_09_PegTransfer.avi')
    vidcap = cv2.VideoCapture('E:\\inf_master\\Semester-5\\Komplexpraktikum\\code\\video\\Course2_Trial_3_28_Circle.avi')
    
    success,image = vidcap.read()
    count = 0
    while success:
        #cv2.imwrite("E:\\inf_master\\Semester-5\\Komplexpraktikum\\code\\video\\frames_1/frame%d.png" % count, image)     # save frame as JPEG file      
        #cv2.imwrite("E:\\inf_master\\Semester-5\\Komplexpraktikum\\code\\video\\frames_2/frame%d.png" % count, image)     # save frame as JPEG file      
        #cv2.imwrite("E:\\inf_master\\Semester-5\\Komplexpraktikum\\code\\video\\frames_3/frame%d.png" % count, image)     # save frame as JPEG file      
        cv2.imwrite("E:\\inf_master\\Semester-5\\Komplexpraktikum\\code\\video\\frames_4/frame%d.png" % count, image)     # save frame as JPEG file      
        
        success,image = vidcap.read()
        print('Read a new frame: ', count)
        count += 1
  


if __name__ == '__main__':
    video2frames()
    '''
    frames2video('/media/huxi/DATA/inf_master/Semester-5/Komplexpraktikum/code/video/result_frame1/')
    '''
