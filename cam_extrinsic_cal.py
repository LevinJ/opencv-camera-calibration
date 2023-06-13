"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Tue Jun 13 2023
*  File : cam_extrinsic_cal.py
******************************************* -->

"""
from poseinfo import PoseInfo, ypr2R
import math
import yaml
import os

class App(object):
    def __init__(self):
        return
    def get_Tvb(self):
        Rvb = [[0, 1, 0],
               [1, 0, 0],
               [0, 0, -1]]
        tvb = [3.01 + 1.71, -0.6, 0]
        Tvb = PoseInfo().construct_fromRt(Rvb, tvb)
        return Tvb
    def get_Tcb(self):
        Rbc= (179.697032, -0.989637, 75.201500)
        tcb = (0.637253, 0.441994, 1.011266)




        Rbc = ypr2R(Rbc)
        Rcb = Rbc.transpose()

        Tcb = PoseInfo().construct_fromRt(Rcb, tcb)
        return Tcb
    def save_yaml(self, yaml_file, Tvc):
        y = {}
        y["x"], y['y'] , y['z'] = Tvc.t.tolist()

        y['qx'], y['qy'],y['qz'],y['qw'] = Tvc.q.tolist()
        y['yaw'], y['pitch'], y['roll'] = (Tvc.ypr * 180/math.pi).tolist()

        with open(yaml_file, 'w') as file:
            outputs = yaml.dump(y, file)
        print('saved extrinsic {}'.format(yaml_file))
        return
    def run(self):
        home_directory = os.path.expanduser( '~' )
        image_dir = '{}/temp/0612'.format(home_directory)
        yaml_file = '{}/extrinsic.yaml'.format(image_dir)
        Tcb = self.get_Tcb()
        Tvb = self.get_Tvb()
        Tvc = Tvb * Tcb.I
        self.save_yaml(yaml_file, Tvc)
        print("Tvc = {}".format(Tvc))
        print('q={}'.format(Tvc.q))



        
        return 

if __name__ == "__main__":   
    obj= App()
    obj.run()
