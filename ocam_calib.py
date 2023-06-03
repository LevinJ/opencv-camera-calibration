"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Sat Jun 03 2023
*  File : ocam_calib.py
******************************************* -->

"""
from camera_calib import CameraCalib
import cv2
import numpy as np

class App(CameraCalib):
    def __init__(self):
        return
    
    def undistort(self, img):
        h, w = self.h, self.w
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coefs, (w, h), 1, (w, h))
        newcameramtx = self.camera_matrix
        roi = None
        dst = cv2.fisheye.undistortImage(img, self.camera_matrix, self.dist_coefs, None, newcameramtx)
        return dst, roi
    def calibrate(self):
        obj_points = np.expand_dims(np.asarray(self.obj_points), -2).astype(np.float64)
        img_points = np.asarray(self.img_points).astype(np.float64)
        calibration_flags = 0
        N_OK = len(obj_points)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        xi = np.zeros(1)

        critia = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 0.0001)
        #probalematic so far, https://github.com/opencv/opencv_contrib/issues/3161
        #https://github.com/opencv/opencv_contrib/issues/804
        self.rms, self.camera_matrix, self.xi,  self.dist_coefs, self.rvecs, self.tvecs, self.idx = \
            cv2.omnidir.calibrate(obj_points, img_points, (self.w, self.h), K, 
                                  xi, D, calibration_flags, critia)
        return
    def reproject(self, obj_points, rvec, tvec):
        obj_points = np.expand_dims(np.asarray(obj_points), -2)

        img_points2, _ = cv2.fisheye.projectPoints(obj_points, rvec, tvec, self.camera_matrix, self.xi, self.dist_coefs) 
        return img_points2
    def run(self):

        image_dir = './canon-efs-24mm-crop1.6'
       
        


        #<w>x<h>              Number of *inner* corners of the chessboard pattern (default: 9x6)
        self.corners = (9, 6)
        #<w>x<h>  Physical sensor size in mm (optional)
        self.sensor_size = (22.3, 14.9)
        #Square size in m
        self.square_size = 0.0244
        #Number of threads to use
        self.threads = 8
        file_patter = "*.JPG"
        self.start_calib(image_dir, file_patter)
        return 

if __name__ == "__main__":   
    obj= App()
    obj.run()
