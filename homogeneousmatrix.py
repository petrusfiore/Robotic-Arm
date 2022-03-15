#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Functions for working with homogeneous matrices. """

__author__ = ("Guénhaël Le Quilliec")

# TODO s'inspirer de ToPy/core/data/H8_K.py pour créer des matrices scipy

from numpy import array, zeros, sin, cos, dot, arctan2, argsort, absolute, cross, eye
from numpy import abs as npabs
from numpy.linalg import norm, det

tol = 1e-9

# Rx
# Ry
# Rz
# Rxy
# Ryx
# Ryz
# Rzx
# Rzy
# Rxyz
# Rzyx

# Tx
# Ty
# Tz
# Txy
# Txz
# Tyz
# Txyz

# TzRz
# TzxRy
# TzRzyx
# TzxRxzy
# TxyzRzyx
# TzxyRyxz

# zaligned (use zyaligned instead)
# zyaligned

# ishomogeneousmatrix
# pdot
# vdot
# inv
# adjoint
# iadjoint
# Rzyx_angles

# adjacency_tw
# exp_tw

######################################################################

def Rx(q1):
    """ Homogeneous matrix of a rotation around the x-axis.

    :param float angle: angle around x-axis in radian
    :return: homogeneous matrix
    :rtype: (4,4)-array

    **Example:**

    >>> Rx(3.14/6)
    array([[ 1.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.86615809, -0.4997701 ,  0.        ],
           [ 0.        ,  0.4997701 ,  0.86615809,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])

    """
    c1 = cos(q1)
    s1 = sin(q1)
    H = array(
        [[1., 0.,  0., 0.],
         [0., c1, -s1, 0.],
         [0., s1,  c1, 0.],
         [0., 0.,  0., 1.]])
    return H

def Ry(q1):
    """ Homogeneous matrix of a rotation around the y-axis.

    :param float angle: angle around y-axis in radian
    :return: homogeneous matrix
    :rtype: (4,4)-array

    **Example:**

    >>> Ry(3.14/6)
    array([[ 0.86615809,  0.        ,  0.4997701 ,  0.        ],
           [ 0.        ,  1.        ,  0.        ,  0.        ],
           [-0.4997701 ,  0.        ,  0.86615809,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])

    """
    c1 = cos(q1)
    s1 = sin(q1)
    H = array(
        [[ c1, 0., s1, 0.],
         [ 0., 1., 0., 0.],
         [-s1, 0., c1, 0.],
         [ 0., 0., 0., 1.]])
    return H

def Rz(q1):
    """ Homogeneous matrix of a rotation around the z-axis.

    :param float angle: angle around z-axis in radian
    :return: homogeneous matrix
    :rtype: (4,4)-array

    **Example:**

    >>> Rz(3.14/6)
    array([[ 0.86615809, -0.4997701 ,  0.        ,  0.        ],
           [ 0.4997701 ,  0.86615809,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  1.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])

    """
    c1 = cos(q1)
    s1 = sin(q1)
    H = array(
        [[c1, -s1, 0., 0.],
         [s1,  c1, 0., 0.],
         [0.,  0., 1., 0.],
         [0.,  0., 0., 1.]])
    return H

def Rxy(q1, q2):
    """ Homogeneous transformation matrix from roll-pitch angles.

    :param float Rx: roll angle in radian
    :param float Ry: pitch angle in radian
    :return: homogeneous matrix of the roll-pitch orientation
    :rtype: (4,4)-array
    
    In short, return: \R = \R_{x} * \R_{y}

    """
    c1 = cos(q1)
    s1 = sin(q1)
    c2 = cos(q2)
    s2 = sin(q2)
    return array(
        [[   c2 , 0.,     s2 , 0.],
         [ s1*s2, c1,  -s1*c2, 0.],
         [-c1*s2, s1,   c1*c2, 0.],
         [   0. , 0.,     0. , 1.]])

def Ryx(q1, q2):
    """ Homogeneous transformation matrix from roll-pitch angles.

    :param float Ry: pitch angle in radian
    :param float Rx: roll angle in radian
    :return: homogeneous matrix of the roll-pitch orientation
    :rtype: (4,4)-array
    
    In short, return: \R = \R_{y} * \R_{x}

    **Example:**

    >>> Ryx(3.14/4, 3.14/3)
    array([[ 0.70738827,  0.61194086,  0.35373751,  0.        ],
           [ 0.        ,  0.50045969, -0.86575984,  0.        ],
           [-0.70682518,  0.61242835,  0.35401931,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])

    """
    c1 = cos(q1)
    s1 = sin(q1)
    c2 = cos(q2)
    s2 = sin(q2)
    return array(
        [[ c1, s1*s2, s1*c2, 0.],
         [ 0.,   c2 ,  -s2 , 0.],
         [-s1, c1*s2, c1*c2, 0.],
         [ 0.,   0. ,   0. , 1.]])

def Ryz(q1, q2):
    """ Homogeneous transformation matrix from roll-pitch angles.

    :param float Ry: pitch angle in radian
    :param float Rz: roll angle in radian
    :return: homogeneous matrix of the roll-pitch orientation
    :rtype: (4,4)-array
    
    In short, return: \R = \R_{y} * \R_{z}

    """
    c1 = cos(q1)
    s1 = sin(q1)
    c2 = cos(q2)
    s2 = sin(q2)
    return array(
        [[ c1*c2, -c1*s2, s1, 0.],
         [   s2 ,    c2 , 0., 0.],
         [-c2*s1,  s1*s2, c1, 0.],
         [   0. ,    0. , 0., 1.]])

def Rzx(q1, q2):
    """ Homogeneous transformation matrix from roll-yaw angles.

    :param float Rz: yaw angle in radian
    :param float Rx: roll angle in radian
    :return: homogeneous matrix of the roll-yaw orientation
    :rtype: (4,4)-array
    
    In short, return: \R = \R_{z} * \R_{x}

    **Example:**

    >>> Rzx(3.14/6, 3.14/3)
    array([[ 0.86615809, -0.25011479,  0.43268088,  0.        ],
           [ 0.4997701 ,  0.43347721, -0.74988489,  0.        ],
           [ 0.        ,  0.86575984,  0.50045969,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])

    """
    c1 = cos(q1)
    s1 = sin(q1)
    c2 = cos(q2)
    s2 = sin(q2)
    return array(
        [[ c1,-s1*c2, s1*s2, 0.],
         [ s1, c1*c2,-c1*s2, 0.],
         [ 0., s2   , c2   , 0.],
         [ 0., 0.   , 0.   , 1.]])

def Rzy(q1, q2):
    """ Homogeneous transformation matrix from pitch-yaw angles.

    :param float Rz: yaw angle in radian
    :param float Ry: pitch angle in radian
    :return: homogeneous matrix of the pitch-yaw orientation
    :rtype: (4,4)-array
    
    In short, return: \R = \R_{z} * \R_{y}

    Example:

    >>> Rzy(3.14/6, 3.14/4)
    array([[ 0.61271008, -0.4997701 ,  0.61222235,  0.        ],
           [ 0.35353151,  0.86615809,  0.35325009,  0.        ],
           [-0.70682518,  0.        ,  0.70738827,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])

    """
    c1 = cos(q1)
    s1 = sin(q1)
    c2 = cos(q2)
    s2 = sin(q2)
    return array(
        [[ c1*c2,-s1, c1*s2, 0.],
         [ s1*c2, c1, s1*s2, 0.],
         [-s2   , 0., c2   , 0.],
         [ 0.   , 0., 0.   , 1.]])

def Rxyz(q1, q2, q3):
    """
    """
    c1 = cos(q1)
    s1 = sin(q1)
    c2 = cos(q2)
    s2 = sin(q2)
    c3 = cos(q3)
    s3 = sin(q3)
    return array(
        [[ c2*c3         ,-c2*s3         , s2   , 0.],
         [ c1*s3+c3*s1*s2, c1*c3-s1*s2*s3,-c2*s1, 0.],
         [ s1*s3-c1*c3*s2, c1*s2*s3+c3*s1, c1*c2, 0.],
         [ 0.            , 0.            , 0.   , 1.]])

def Rzyx(q1, q2, q3):
    """ Homogeneous transformation matrix from roll-pitch-yaw angles.

    :param float q1: yaw angle in radian
    :param float q2: pitch angle in radian
    :param float q3: roll angle in radian
    :return: homogeneous matrix of the roll-pitch-yaw orientation
    :rtype: (4,4)-array
    
    In short, return: \R = \R_{z} * \R_{y} * \R_{x}

    **Example:**

    >>> Rzyx(3.14/6, 3.14/4, 3.14/3)
    array([[ 0.61271008,  0.27992274,  0.73907349,  0.        ],
           [ 0.35353151,  0.73930695, -0.57309746,  0.        ],
           [-0.70682518,  0.61242835,  0.35401931,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])

    """
    c1 = cos(q1)
    s1 = sin(q1)
    c2 = cos(q2)
    s2 = sin(q2)
    c3 = cos(q3)
    s3 = sin(q3)
    return array(
        [[ c1*c2, c1*s2*s3-s1*c3, c1*s2*c3+s1*s3, 0.],
         [ s1*c2, s1*s2*s3+c1*c3, s1*s2*c3-c1*s3, 0.],
         [-s2   , c2*s3         , c2*c3         , 0.],
         [ 0.   , 0.            , 0.            , 1.]])
         
######################################################################

def Tx(q1):
    """
    """
    return array(
        [[ 1. , 0., 0., q1],
         [ 0. , 1., 0., 0.],
         [ 0. , 0., 1., 0.],
         [ 0. , 0., 0., 1.]])

def Ty(q1):
    """
    """
    return array(
        [[ 1. , 0., 0., 0.],
         [ 0. , 1., 0., q1],
         [ 0. , 0., 1., 0.],
         [ 0. , 0., 0., 1.]])

def Tz(q1):
    """
    """
    return array(
        [[ 1. , 0., 0., 0.],
         [ 0. , 1., 0., 0.],
         [ 0. , 0., 1., q1],
         [ 0. , 0., 0., 1.]])

def Txy(q1, q2):
    """
    """
    return array(
        [[ 1. , 0., 0., q1],
         [ 0. , 1., 0., q2],
         [ 0. , 0., 1., 0.],
         [ 0. , 0., 0., 1.]])

def Txz(q1, q2):
    """
    """
    return array(
        [[ 1. , 0., 0., q1],
         [ 0. , 1., 0., 0.],
         [ 0. , 0., 1., q2],
         [ 0. , 0., 0., 1.]])

def Tyz(q1, q2):
    """
    """
    return array(
        [[ 1. , 0., 0., 0.],
         [ 0. , 1., 0., q1],
         [ 0. , 0., 1., q2],
         [ 0. , 0., 0., 1.]])

def Txyz(q1, q2, q3):
    """ Homogeneous matrix of a translation.

    :param float Tx, Ty, Tz: coordinates of the translation vector in 3d space
    :return: homogeneous matrix of the translation
    :rtype: (4,4)-array

    **Example:**

    >>> Txyz(1., 2., 3.)
    array([[ 1.,  0.,  0.,  1.],
           [ 0.,  1.,  0.,  2.],
           [ 0.,  0.,  1.,  3.],
           [ 0.,  0.,  0.,  1.]])

    """
    return array(
        [[ 1. , 0., 0., q1],
         [ 0. , 1., 0., q2],
         [ 0. , 0., 1., q3],
         [ 0. , 0., 0., 1.]])

######################################################################

def TzRz(q1, q2):
    """
    """
    c2 = cos(q2)
    s2 = sin(q2)
    return array(
        [[c2,-s2, 0., 0.],
         [s2, c2, 0., 0.],
         [0., 0., 1., q1],
         [0., 0., 0., 1.]])

def TzxRy(q1, q2, q3):
    """
    """
    c3 = cos(q3)
    s3 = sin(q3)
    return array(
        [[ c3, 0., s3, q2],
         [ 0., 1., 0., 0.],
         [-s3, 0., c3, q1],
         [ 0., 0., 0., 1.]])

def TzRzyx(q1, q2, q3, q4):
    """
    """
    c2 = cos(q2)
    s2 = sin(q2)
    c3 = cos(q3)
    s3 = sin(q3)
    c4 = cos(q4)
    s4 = sin(q4)
    return array(
        [[c2*c3, c2*s3*s4-c4*s2, s2*s4+c2*c4*s3, 0.],
         [c3*s2, s2*s3*s4+c2*c4, c4*s2*s3-c2*s4, 0.],
         [ -s3 ,     c3*s4     ,     c3*c4     , q1],
         [  0. ,       0.      ,       0.      , 1.]])

def TzxRxzy(q1, q2, q3, q4, q5):
    """
    """
    c3 = cos(q3)
    s3 = sin(q3)
    c4 = cos(q4)
    s4 = sin(q4)
    c5 = cos(q5)
    s5 = sin(q5)
    return array([[c4*c5,-s4,c4*s5,q2],[s3*s5+c3*c5*s4,c3*c4,c3*s4*s5-c5*s3,0],[c5*s3*s4-c3*s5,c4*s3,s3*s4*s5+c3*c5,q1],[0,0,0,1]])

def TxyzRzyx(q1, q2, q3, q4, q5, q6):
    """
    """
    c2 = cos(q4)
    s2 = sin(q4)
    c3 = cos(q5)
    s3 = sin(q5)
    c4 = cos(q6)
    s4 = sin(q6)
    return array(
        [[c2*c3, c2*s3*s4-c4*s2, s2*s4+c2*c4*s3, q1],
         [c3*s2, s2*s3*s4+c2*c4, c4*s2*s3-c2*s4, q2],
         [ -s3 ,     c3*s4     ,     c3*c4     , q3],
         [  0. ,       0.      ,       0.      , 1.]])

def TzxyRyxz(q1, q2, q3, q4, q5, q6):
    """
    """
    c4 = cos(q4)
    s4 = sin(q4)
    c5 = cos(q5)
    s5 = sin(q5)
    c6 = cos(q6)
    s6 = sin(q6)
    return array([[s4*s5*s6+c4*c6,c6*s4*s5-c4*s6,c5*s4,q2],[c5*s6,c5*c6,-s5,q3],[c4*s5*s6-c6*s4,s4*s6+c4*c6*s5,c4*c5,q1],[0,0,0,1]])

"""
def zaligned(vec):
    " "" Returns an homogeneous matrix whose z-axis is colinear with *vec*.

    :param vec: input vector
    :type  vec: (3,)-array
    :return: homogeneous matrix of the frame aligned with vec
    :rtype: (4,4)-array

    **Example:**

    >>> zaligned((1.,0.,0.))
    array([[-0.,  0.,  1.,  0.],
           [ 0., -1.,  0.,  0.],
           [ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.]])

    " ""
    # Initialization
    H = eye(4)
    x = H[0:3, 0]
    y = H[0:3, 1]
    z = H[0:3, 2]

    # z axis
    z[:] = vec/norm(vec)

    # y axis, normal to z-axis
    i = argsort(absolute(z))
    y[i] = (0, z[i[2]], -z[i[1]])
    y /= norm(y)
    
    # x axis
    x[:] = cross(y,z)
    return H
"""

def zyaligned(vecz, vecy=None):
    """
    """
    # Initialization
    H = eye(4)
    x = H[0:3, 0]
    y = H[0:3, 1]
    z = H[0:3, 2]
    # z axis
    z[:] = vecz/norm(vecz)
    # x axis
    if vecy == None:
        i = argsort(absolute(z))
        x[i] = (0, z[i[2]], -z[i[1]])
    else:
        x[:] = cross(vecy, vecz)
    x /= norm(x)
    # y axis
    y[:] = cross(z,x)
    return H

def ishomogeneousmatrix(H, _tol=tol):
    """ Return true if input is an homogeneous matrix.

    :param H: the homogeneous matrix to check
    :type  H: (4,4)-array
    :param float _tol: the tolerance for the rotation matrix determinant
    :return: True if homogeneous matrix, False otherwise

    """
    return (H.shape == (4, 4)) and (npabs(det(H[0:3, 0:3])-1) <= _tol) and (H[3, 0:4]==[0, 0, 0, 1]).all()

def pdot(H, point):
    r""" Frame displacement for a point.

    :param H: the homogeneous matrix to check
    :type  H: (4,4)-array
    :param point: point one wants to displace
    :type  point: (3,)-array
    :return: the displaced point
    :rtype: (3,)-array

    `\pt{a} = \H_{ab}  \pt{b}`, where `\H_{ab}` is the
    homogeneous matrix from `\Frame{a}` to `\Frame{b}`, and `\pt{b}` is
    the point expressed in `\Frame{b}`.

    """
    assert ishomogeneousmatrix(H)
    return dot(H[0:3, 0:3], point) + H[0:3, 3]

def vdot(H, vec):
    r""" Frame displacement for a vector.

    :param H: the homogeneous matrix to check
    :type  H: (4,4)-array
    :param vec: point one wants to displace
    :type  vec: (3,)-array
    :return: the displaced point
    :rtype: (3,)-array

    `\ve{a} = \R_{ab}  \ve{b}`, where `\R_{ab}` is the
    rotation matrix from `\Frame{a}` to `\Frame{b}`, and `\ve{b}` is
    the vector expressed in `\Frame{b}`.

    """
    assert ishomogeneousmatrix(H)
    return dot(H[0:3, 0:3], vec)

def inv(H):
    """ Invert a homogeneous matrix.

    :param H: the homogeneous matrix to invert
    :type  H: (4,4)-array
    :return: inverted homogeneous matrix
    :rtype: (4,4)-array

    **Example:**

    >>> H = array(
    ...     [[ 0.70738827,  0.        , -0.70682518,  3.        ],
    ...      [ 0.61194086,  0.50045969,  0.61242835,  4.        ],
    ...      [ 0.35373751, -0.86575984,  0.35401931,  5.        ],
    ...      [ 0.        ,  0.        ,  0.        ,  1.        ]])
    >>> inv(H)
    array([[ 0.70738827,  0.61194086,  0.35373751, -6.3386158 ],
           [ 0.        ,  0.50045969, -0.86575984,  2.32696044],
           [-0.70682518,  0.61242835,  0.35401931, -2.09933441],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])

    """
    assert ishomogeneousmatrix(H)
    R = H[0:3, 0:3]
    p = H[0:3, 3]
    
    invH = zeros((4,4))
    invH[0:3, 0:3] = R.T
    invH[0:3,3]    = -dot(R.T, p)
    invH[3,3]      = 1.
    return invH

def adjoint(H):
    """ Adjoint of the homogeneous matrix.

    :param H: homogeneous matrix
    :type  H: (4,4)-array
    :return: adjoint matrix
    :rtype: (6,6)-array

    **Example:**

    >>> H = array(
    ...     [[ 0.70738827,  0.        , -0.70682518,  3.        ],
    ...      [ 0.61194086,  0.50045969,  0.61242835,  4.        ],
    ...      [ 0.35373751, -0.86575984,  0.35401931,  5.        ],
    ...      [ 0.        ,  0.        ,  0.        ,  1.        ]])
    >>> adjoint(H)
    array([[ 0.70738827,  0.        , -0.70682518,  0.        ,  0.        ,
             0.        ],
           [ 0.61194086,  0.50045969,  0.61242835,  0.        ,  0.        ,
             0.        ],
           [ 0.35373751, -0.86575984,  0.35401931,  0.        ,  0.        ,
             0.        ],
           [-1.64475426, -5.96533781, -1.64606451,  0.70738827,  0.        ,
            -0.70682518],
           [ 2.47572882,  2.59727952, -4.59618383,  0.61194086,  0.50045969,
             0.61242835],
           [-0.9937305 ,  1.50137907,  4.66458577,  0.35373751, -0.86575984,
             0.35401931]])

    """
    assert ishomogeneousmatrix(H)
    R = H[0:3, 0:3]
    p = H[0:3, 3]
    pxR = dot(
        array(
            [[    0, -p[2],  p[1]],
             [ p[2],     0, -p[0]],
             [-p[1],  p[0],     0]]),
        R)

    Ad = zeros((6,6))
    Ad[0:3,0:3] = R
    Ad[3:6,0:3] = pxR
    Ad[3:6,3:6] = R
    return Ad


def iadjoint(H):
    """ Return the adjoint ((6,6) array) of the inverse homogeneous matrix.
    """
    return adjoint(inv(H))


def dAdjoint(Ad, T):
    """ Return the derivative of an Adjoint with respect to time.

    :param Ad: the Adjoint matrix one wants the derivative.
    :type  Ad: (6,6)-array
    :param T:  the corresponding twist
    :type  T: (6,)-array
    :return: the derivative of the adjoint matrix
    :rtype:  (6,6)-array

    **Definition from ope-matlab:**

    if H is defined as follow:
    
    .. math::

        x{a} = H * x{b}
        Ad = adjoint( H )
        T  = velocity of {b} relative to {a} expressed in {b}

    """
    return dot(Ad, adjacency_tw(T))

# TODO use from scipy.spatial.transform import Rotation
def Rzyx_angles(H):
    """ Returns the roll-pitch-yaw angles corresponding to the rotation matrix of `\H`.

    :param H: homogeneous matrix
    :type  H: (4,4)-array
    :return: angles of roll pitch yaw
    :rtype: (3,)-array

    Returns the angles such that `H[0:3, 0:3] = R_z(a_z) R_y(a_y) R_x(a_x)`.

    **Example:**

    >>> angles = array((3.14/3, 3.14/6, 1))
    >>> (Rzyx_angles(Rzyx(*angles)) == angles).all()
    True

    """
    assert ishomogeneousmatrix(H)
    if abs(H[0, 0]) < tol and abs(H[1, 0]) < tol:
        # singularity
        az = 0.0
        ay = arctan2(-H[2, 0], H[0, 0])
        ax = arctan2(-H[1, 2], H[1, 1])
    else:
        az = arctan2( H[1, 0], H[0, 0])
        sz = sin(az)
        cz = cos(az)
        ay = arctan2(-H[2, 0], cz*H[0, 0] + sz*H[1, 0])
        ax = arctan2(sz*H[0, 2] - cz*H[1, 2], cz*H[1, 1] - sz*H[0, 1])
    return array(az, ay, ax)

######################################################################

# author: Sébastien BARTHÉLEMY
def adjacency_tw(tw):
    r"""
    Return the adjacency matrix.

    :param tw: a twist to build the adjacency matrix.
    :type  tw: (6,)-array
    :rtype: (6,6)-array
    """

    # TODO
    """
    **Example:**

    >>> t = array([1., 2., 3., 10., 11., 12.])
    >>> adjacency(t)
    array([[  0.,  -3.,   2.,   0.,   0.,   0.],
           [  3.,   0.,  -1.,   0.,   0.,   0.],
           [ -2.,   1.,   0.,   0.,   0.,   0.],
           [  0., -12.,  11.,   0.,  -3.,   2.],
           [ 12.,   0., -10.,   3.,   0.,  -1.],
           [-11.,  10.,   0.,  -2.,   1.,   0.]])

    """
    assert tw.shape == (6,)
    return array(
        [[     0,-tw[2], tw[1],      0,     0,     0],
         [ tw[2],     0,-tw[0],      0,     0,     0],
         [-tw[1], tw[0],     0,      0,     0,     0],
         [     0,-tw[5], tw[4],      0,-tw[2], tw[1]],
         [ tw[5],     0,-tw[3],  tw[2],     0,-tw[0]],
         [-tw[4], tw[3],     0, -tw[1], tw[0],     0]])

# author: Sébastien BARTHÉLEMY
def exp_tw(tw):
    r"""
    
    Return the exponential of the twist matrix.

    :param tw: a twist to build the adjacency matrix.
    :type  tw: (6,)-array
    :return: a homogeneous matrix corresponding to the displacement generated by the twist in 1 second.
    :rtype: (4,4)-array
    """
    
    # TODO
    """
    **Example:**

    >>> t = array([1., 2., 3., 10., 11., 12.])
    >>> exp(t)
    array([[ -0.69492056,   0.71352099,   0.08929286,   2.90756949],
           [ -0.19200697,  -0.30378504,   0.93319235,  11.86705709],
           [  0.69297817,   0.6313497 ,   0.34810748,  13.78610544],
           [  0.        ,   0.        ,   0.        ,   1.        ]])

    """
    assert tw.shape == (6,)
    w = tw[0:3]
    v = tw[3:6]
    wx = array(
        [[     0,-tw[2], tw[1]],
         [ tw[2],     0,-tw[0]],
         [-tw[1], tw[0],     0]])
    t = norm(w)
    if t >= 0.001:
        cc = (1-cos(t))/t**2
        sc = sin(t)/t
        dsc = (t-sin(t))/t**3
    else:
        cc = 1./2.
        sc = 1.-t**2/6.
        dsc = 1./6.

    R = eye(3) + sc*wx + cc*dot(wx, wx)
    p = dot(sc*eye(3) + cc*wx + dsc*outer(w,w), v)

    H = zeros((4,4))
    H[0:3,0:3] = R
    H[0:3,3] = p
    H[3,3] = 1

    return H
