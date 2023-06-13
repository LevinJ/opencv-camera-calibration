"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Tue Jun 13 2023
*  File : cam_extrinsic_cal.py
******************************************* -->

"""
from poseinfo import PoseInfo, ypr2R

class App(object):
    def __init__(self):
        return
    def get_Tvb(self):
        Rvb = [[-1, 0, 0],
               [0, 1, 0],
               [0, 0, -1]]
        tvb = [2.65 + 3, 0.612147, 0]
        Tvb = PoseInfo().construct_fromRt(Rvb, tvb)
        return Tvb
    def run(self):
        Rbc= (-91.074362, 0.170404, 75.477268)
        tcb = (-0.612147, 0.218198, 1.823270)




        Rbc = ypr2R(Rbc)
        Rcb = Rbc.transpose()

        Tcb = PoseInfo().construct_fromRt(Rcb, tcb)

        Tvb = self.get_Tvb()

        Tvc = Tvb * Tcb.I
        print("Tvc = {}".format(Tvc))



        
        return 

if __name__ == "__main__":   
    obj= App()
    obj.run()
