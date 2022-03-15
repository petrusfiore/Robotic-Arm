from frankx import *

import homogeneousmatrix as hm

from math import pi
from numpy import array, angle

robot = Robot('192.168.2.1')
robot.set_dynamic_rel(0.05)

motion = LinearRelativeMotion(Affine(0.0, 0.2, 0.0))

robot.move(motion)

motion = LinearRelativeMotion(Affine(0.0, -0.2, 0.0))

robot.move (motion)


robot.set_default_behavior()

# https://1drv.ms/u/s!An5u4x5k5YN2gudBfgr9cGiaJzL3KQ?e=EnPsRQ