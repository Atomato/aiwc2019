# Author(s): Luiz Felipe Vecchietti, Chansol Hong, Inbae Jeong
# Maintainer: Chansol Hong (cshong@rit.kaist.ac.kr)

import math

def distance(x1, x2, y1, y2):
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

def distance2(x1, x2, y1, y2):
    return math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2)    

def degree2radian(deg):
    return deg * math.pi / 180

def radian2degree(rad):
    return rad * 180 / math.pi

def dipole_potential(x, y, a, b):
	V = b*(- 1/distance(x, -a, y, 0) + 1/distance(x, a, y, 0))
	return V

def rot_transform(x_center, y_center, theta, x, y):
	x_new = (x-x_center)*math.cos(theta) - (y-y_center)*math.sin(theta)
	y_new = (x-x_center)*math.sin(theta) + (y-y_center)*math.cos(theta)
	return [x_new, y_new]