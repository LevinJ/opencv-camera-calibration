"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Fri Jun 02 2023
*  File : fisheye_calib.py
******************************************* -->

"""
from camera_calib import CameraCalib
import cv2
import numpy as np

class App(CameraCalib):
    def __init__(self):
        CameraCalib.__init__(self)
        return
    def undistort(self, img):
        h, w = self.h, self.w
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coefs, (w, h), 1, (w, h))
        newcameramtx = self.camera_matrix
        roi = None
        dst = cv2.fisheye.undistortImage(img, self.camera_matrix, self.dist_coefs, None, newcameramtx)
        return dst, roi
    def calibrate(self):
        obj_points = np.expand_dims(np.asarray(self.obj_points), 1)
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
        N_OK = len(obj_points)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        self.rms, self.camera_matrix, self.dist_coefs, self.rvecs, self.tvecs = \
            cv2.fisheye.calibrate(obj_points, self.img_points, (self.w, self.h), K,
                                D,
                                rvecs,
                                tvecs,
                                calibration_flags,
                                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)) 
        return
    def reproject(self, obj_points, rvec, tvec):
        obj_points = np.expand_dims(np.asarray(obj_points), -2)
        img_points2, _ = cv2.fisheye.projectPoints(obj_points, rvec, tvec, self.camera_matrix, self.dist_coefs) 
        return img_points2
    def run(self):
        image_dir = '/home/levin/temp/0601'
        #<w>x<h>              Number of *inner* corners of the chessboard pattern (default: 9x6)
        self.corners = (11, 8)
        #<w>x<h>  Physical sensor size in mm (optional)
        self.sensor_size = (22.3, 14.9)
        #Square size in m
        self.square_size = 0.0244
        #Number of threads to use
        self.threads = 10
        file_patter = "*.jpg"
        self.start_calib(image_dir, file_patter)
        return 

if __name__ == "__main__":   
    obj= App()
    obj.run()
