"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Fri Jun 02 2023
*  File : camera_calib.py.py
******************************************* -->

"""
import json, os
import numpy as np
import cv2
from glob import glob
import sys
import math
from math import  atan2
from math import sin
from math import cos

def euler_from_matrix(R):
    R = np.array(R)
    n = R[:, 0]
    o = R[:, 1]
    a =R[:, 2]

    y = atan2(n[1], n[0])
    p = atan2(-n[2], n[0] * cos(y) + n[1] * sin(y))
    r = atan2(a[0] * sin(y) - a[1] * cos(y), -o[0] * sin(y) + o[1] * cos(y))
    ypr = np.array([y, p, r])
    return ypr
def R2ypr(R):
    return euler_from_matrix(R)/math.pi * 180

class CameraCalib(object):
    def __init__(self):
        self.do_debug = True
        return
    def splitfn(self, fname):
        path, fname = os.path.split(fname)
        name, ext = os.path.splitext(fname)
        return path, name, ext
    def calibrate(self):
        self.rms, self.camera_matrix, self.dist_coefs, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(self.obj_points, self.img_points, (self.w, self.h), None, None) #, None, None, None)
        return
    def reproject(self, obj_points, rvec, tvec):
        img_points2, _ = cv2.projectPoints(obj_points, rvec, tvec, self.camera_matrix, self.dist_coefs) 
        return img_points2
    def undistort(self, img):
        h, w = self.h, self.w
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coefs, (w, h), 1, (w, h))

        dst = cv2.undistort(img, self.camera_matrix, self.dist_coefs, None, newcameramtx)
        return dst, roi
    def save_yaml(self, j, json_file):
        return
    def calibrate_reproject(self, obj_points, img_points, w, h, num_chessboards, cb_to_image_index, image_files, j):
        # Calculate camera matrix, distortion, etc
        self.obj_points = obj_points
        self.img_points = img_points
        self.w, self.h = w, h
        
        self.calibrate()
        print("RMS:", self.rms)
        print()
        
        rms, camera_matrix, dist_coefs, rvecs, tvecs = self.rms, self.camera_matrix, self.dist_coefs, self.rvecs, self.tvecs
        # Compute reprojection error
        # After https://docs.opencv2.org/4.5.2/dc/dbb/tutorial_py_calibration.html
        print('Computing reprojection error:')
        reprod_error = {}
        errors = []
        for cb_index in range(num_chessboards):      
            img_points2 = self.reproject(obj_points[cb_index], rvecs[cb_index], tvecs[cb_index],)
            error = cv2.norm(img_points[cb_index], img_points2, cv2.NORM_L2) / len(img_points2)
            img_index = cb_to_image_index[cb_index]
            img_file = image_files[img_index]
            img_file = os.path.basename(img_file)
            print('[%s] %.6f' % (img_file, error))
            reprod_error[img_file] = error
            errors.append(error)
        reprojection_error_avg = np.average(errors)
        reprojection_error_stddev = np.std(errors)
        print("Average reprojection error: %.6f +/- %.6f" % (reprojection_error_avg, reprojection_error_stddev))
        print()
        print("Camera matrix:\n", camera_matrix)    
        print("Distortion coefficients:", dist_coefs.ravel())
        
        j['camera_matrix'] = camera_matrix.tolist()
        j['distortion_coefficients'] = dist_coefs.ravel().tolist()
        j['rms'] = rms
        j['reprojection_error'] = {'average': reprojection_error_avg, 'stddev': reprojection_error_stddev, 'image': reprod_error }
        return camera_matrix,dist_coefs, rvecs,tvecs
    def main(self, image_files, pattern_size, square_size, threads, json_file=None, debug_dir=None, sensor_size=None):
        """    
        image_files: list of image file names
        pattern_size: the number of *inner* points! So for a grid of 10x7 *squares* there's 9x6 inner points
        square_size: the real-world dimension of a chessboard square, in meters
        threads: number of threads to use
        json_file: JSON file to write calibration data to 
        debug_dir: if set, the path to which debug images with the detected chessboards are written
        """
        
        # JSON data
        j = {}

        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size    
        
        j['chessboard_points'] = pattern_points.tolist()
        j['chessboard_inner_corners'] = pattern_size
        j['chessboard_spacing_m'] = square_size
            
        # Read first image to get resolution
        # TODO: use imquery call to retrieve results
        img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
        if img is None:
            print('Failed to read %s to get resolution!' % image_files[0])
            return
            
        h, w = img.shape[:2]
        print('Image resolution %dx%d' % (w, h))    
        
        j['image_resolution'] = (w, h)
        
        # Process all images

        def process_image(fname):
            sys.stdout.write('.')
            sys.stdout.flush()
            
            img = cv2.imread(fname, 0)
            if img is None:
                return (fname, None, 'Failed to load')
                
            if w != img.shape[1] or h != img.shape[0]:
                return (fname, None, "Size %dx%d doesn't match" % (img.shape[1], img.shape[0]))
            
            found, corners = cv2.findChessboardCorners(img, pattern_size)
            if found:
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

            if debug_dir:
                # Write image with detected chessboard overlay
                vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(vis, pattern_size, corners, found)            
                _, name, _ = self.splitfn(fname)
                outfile = os.path.join(debug_dir, name + '_chessboard.png')
                cv2.imwrite(outfile, vis)

            if not found:
                return (fname, None, 'Chessboard not found')

            return (fname, corners, pattern_points)
        
        if threads <= 1:
            sys.stdout.write('Processing images ')
            results = [process_image(fname) for fname in image_files]
        else:
            sys.stdout.write('Processing images using %d threads ' % threads)
            from multiprocessing.dummy import Pool as ThreadPool
            pool = ThreadPool(threads)
            results = pool.map(process_image, image_files)
            
        sys.stdout.write(' done\n')
        sys.stdout.flush()
        print()
        
        # Prepare calibration input

        obj_points = []
        img_points = []        
        cb_index = 0
        cb_to_image_index = {}
        
        # Sort by file name
        results.sort(key = lambda e: e[0])
        for img_index, result in enumerate(results):
            corners = result[1]
            if corners is None:
                print('[%s] FAILED: %s' % (result[0], result[2]))
                continue
            img_points.append(corners)
            obj_points.append(result[2])
            cb_to_image_index[cb_index] = img_index
            cb_index += 1

        num_chessboards = cb_index
            
        print('Found chessboards in %d out of %d images' % (num_chessboards, len(image_files)))
        print()
        
        if num_chessboards == 0:
            print('No chessboards to use!')
            sys.exit(-1)

        #calibrate and reproject
        camera_matrix,dist_coefs, rvecs,tvecs = self.calibrate_reproject(obj_points, img_points, w, h, num_chessboards, cb_to_image_index, image_files, j)
        
        if sensor_size is not None:
            
            fovx, fovy, focal_length, principal_point, aspect_ratio = \
                cv2.calibrationMatrixValues(camera_matrix, (w,h), sensor_size[0], sensor_size[1])
                
            print()            
            print('FOV: %.6f %.6f degrees' % (fovx, fovy))
            print('Focal length: %.6f mm' % focal_length)
            print('Principal point: %.6f %.6f mm' % principal_point)
            print('Aspect ratio: %.6f' % aspect_ratio)
            
            j['sensor_size_mm'] = sensor_size
            j['fov_degrees'] = (fovx, fovy)
            j['focal_length_mm'] = focal_length
            j['principal_point_mm'] = principal_point
            j['aspect_ratio'] = aspect_ratio
        
        print()
        chessboard_orientations = {}
        for cb_index in range(num_chessboards):
            img_index = cb_to_image_index[cb_index]
            r = rvecs[cb_index]
            t = tvecs[cb_index]

            image_name = os.path.basename(image_files[img_index])
            rotation_matrix, _ = cv2.Rodrigues(r)
            ypr = R2ypr(rotation_matrix)
            print('[%s] rotation (%.6f, %.6f, %.6f), translation (%.6f, %.6f, %.6f)' % \
                (image_name, ypr[0], ypr[1], ypr[2], t[0][0], t[1][0], t[2][0]))
                
            
            chessboard_orientations[image_name] = {
                #'rotation_vector': (r[0][0], r[1][0], r[2][0]),
                'rotation_matrix': rotation_matrix.tolist(),
                'translation': (t[0][0], t[1][0], t[2][0])
            }
            
            # OpenCV untransformed camera orientation is X to the right, Y down,
            # Z along the view direction (i.e. right-handed). This aligns X,Y axes
            # of pixels in the image plane with the X,Y axes in camera space.
            # The orientations describe the transform needed to bring a detected 
            # chessboard from its object space into camera space.
            j['chessboard_orientations'] = chessboard_orientations
                
        # Write to JSON
                
        if json_file is not None:
            json.dump(j, open(json_file, 'wt'))
            self.save_yaml(j, json_file)

        # Undistort the image with the calibration
        if debug_dir is not None:
            print('')
            print('Writing undistorted images to %s directory:' % debug_dir)
            
            for fname in image_files:
                _, name, _ = self.splitfn(fname)
                img_found = os.path.join(debug_dir, name + '_chessboard.png')
                outfile1 = os.path.join(debug_dir, name + '_undistorted.png')
                outfile2 = os.path.join(debug_dir, name + '_undistorted_cropped.png')

                img = cv2.imread(img_found)
                if img is None:
                    print("Can't find chessboard image!")
                    continue

                dst, roi = self.undistort(img)
                
                # save uncropped
                cv2.imwrite(outfile1, dst)

                # crop and save the image
                if roi is not None:
                    x, y, w, h = roi
                    dst = dst[y:y+h, x:x+w]            
                    cv2.imwrite(outfile2, dst)
                
                print(fname)
        return
    def start_calib(self, image_dir, file_patter):
        image_files = '{}/{}'.format(image_dir, file_patter)
        debug_dir = '{}/debug'.format(image_dir)
        if not self.do_debug:
            debug_dir = None
        json_file = '{}/calib.json'.format(image_dir)



        if debug_dir and not os.path.isdir(debug_dir):
            os.mkdir(debug_dir)

        image_files = glob(image_files)       
        self.main(image_files, self.corners, self.square_size, self.threads, json_file, debug_dir, self.sensor_size)
        return


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
    obj= CameraCalib()
    obj.run()
