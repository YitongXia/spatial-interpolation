# -- my_code_hw01.py
# -- hw01 GEO1015.2021
# -- [YOUR NAME]
# -- [YOUR STUDENT NUMBER]
# -- [YOUR NAME]
# -- [YOUR STUDENT NUMBER]


# -- import outside the standard Python library are not allowed, just those:
import math
import numpy as np
import scipy.spatial
import startinpy

nodata_value = -9999

import sys
import math
import csv
import random
import json
import time


# -----

def Point2D(list_pts_3d):
    # convert 3D point to 2D:

    coord_list = []
    for item in list_pts_3d:
        coord = (item[0], item[1])
        coord_list.append(coord)
    return coord_list


def bounding_box(list_pts_3d):
    # create bounding box of the raster
    coord_list = Point2D(list_pts_3d)
    min_x, min_y = np.min(coord_list, axis=0)
    max_x, max_y = np.max(coord_list, axis=0)
    bbox = ((min_x, min_y), (max_x, max_y))
    return bbox


def cal_col_row_size(list_pts_3d, jparams):
    # calculate the number of column and row from params.json file.

    cellsize = jparams['cellsize']
    lower_left = bounding_box(list_pts_3d)[0]
    upper_right = bounding_box(list_pts_3d)[1]
    col_size = math.ceil((upper_right[0] - lower_left[0]) / cellsize)
    row_size = math.ceil((upper_right[1] - lower_left[1]) / cellsize)
    return cellsize, row_size, col_size


def cal_raster_center(bbox, nrow, ncol, nrows, cellsize):
    center_x = bbox[0][0] + (ncol + 0.5) * cellsize
    center_y = bbox[0][1] + (nrows - nrow - 0.5) * cellsize
    # y = lowleft[1] + (nrows - cur_row - 0.5) * cellsize
    point = (center_x, center_y)
    return point


def cal_convexhull(list_pts_3d):
    list_pts_2d = Point2D(list_pts_3d)
    hull = scipy.spatial.ConvexHull(list_pts_2d)
    return hull


def if_in_convexhull(point, convexhull):
    return all((np.dot(eq[:-1], point) + eq[-1] <= 1e-8) for eq in convexhull.equations)


def output_file(raster, filename, list_pt_3d, jparams):
    cellsize, nrows, ncols = cal_col_row_size(list_pt_3d, jparams)
    XLLCORNER = bounding_box(list_pt_3d)[0][0]
    YLLCORNER = bounding_box(list_pt_3d)[0][1]

    with open(filename, 'w') as fh:
        fh.write('{} {}{}'.format('NCOLS', ncols, '\n'))
        fh.write('{} {}{}'.format('NROWS', nrows, '\n'))
        fh.write('{} {}{}'.format('XLLCORNER', XLLCORNER, '\n'))
        fh.write('{} {}{}'.format('YLLCORNER', YLLCORNER, '\n'))
        fh.write('{} {}{}'.format('CELLSIZE', cellsize, '\n'))
        fh.write('{} {}{}'.format('NODATA_VALUE', '-9999', '\n'))
        for i in range(nrows):
            for j in range(ncols):
                fh.write('{} '.format(raster[i][j]))
        fh.write('\n')


def nn_interpolation(list_pts_3d, jparams):
    # two methods for computing nn,1 is calculate the collection of all point in the hull,
    # 2 is calculate the point and calculate if it's in the hull

    list_pts_2d = Point2D(list_pts_3d)
    bbox = bounding_box(list_pts_3d)
    cellsize, nrows, ncols = cal_col_row_size(list_pts_3d, jparams)
    raster = np.zeros((nrows, ncols))
    hull = scipy.spatial.ConvexHull(list_pts_2d)
    for i in range(nrows):
        for j in range(ncols):
            kd = scipy.spatial.KDTree(list_pts_2d)
            center = cal_raster_center(bbox, i, j, nrows, cellsize)
            d, index = kd.query(center,p=2, k=1)
            raster_value = list_pts_3d[index][2]
            if if_in_convexhull(center, hull):
                raster[i][j] = raster_value
            else:
                raster[i][j] = nodata_value

    output_file(raster, jparams['output-file'], list_pts_3d, jparams)

    # print("cellsize:", jparams['cellsize'])

    # -- to speed up the nearest neighbour us a kd-tree
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html#scipy.spatial.KDTree
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html#scipy.spatial.KDTree.query
    # kd = scipy.spatial.KDTree(list_pts)
    # d, i = kd.query(p, k=1)

    print("File written to", jparams['output-file'])


def if_in_ellipse(pt, a, b):
    if ((pt[0] / a) ** 2 + (pt[1] / b) ** 2) > 1:
        return False
    else:
        return True


def dis(center, point):
    distance = math.sqrt((center[0] - point[0]) ** 2 + (center[1] - point[1]) ** 2)
    return distance


def get_idw_points(kd, center, radius1, radius2, list_pts_3d):
    weight_pt = []
    a = radius1
    b = radius2
    # points = Point2D(list_pts_3d)
    # kd = scipy.spatial.KDTree(points)
    for index in kd.query_ball_point(center, radius1):
        x, y = list_pts_3d[index][0] - center[0], list_pts_3d[index][1] - center[1]  # in the original coordinate system
        if (x * x) / (a * a) + (y * y) / (b * b) <= 1:
            weight_pt.append(index)
    return weight_pt


def cal_weight(dt, center, kd, radius1, radius2, power, list_pts_3d):
    points = Point2D(list_pts_3d)
    find = scipy.spatial.Delaunay.find_simplex(dt, center)
    if find == -1:
        return nodata_value  # point outside of the tin(outside of the convex hull)
    weight_pt = get_idw_points(kd, center, radius1, radius2, list_pts_3d)
    if len(weight_pt) <= 0:
        return nodata_value  # if no points found return nodata
    else:
        weight_sum = 0
        value_sum = 0
        for pt in weight_pt:
            weight_sum += math.pow(dis(center, points[pt]), -power)
            value_sum += math.pow(dis(center, points[pt]), -power) * list_pts_3d[pt][2]
        return (value_sum / weight_sum) if weight_sum != 0 else 0


def idw_interpolation(list_pts_3d, jparams):
    """
    !!! TO BE COMPLETED !!!

    Function that writes the output raster with IDW

    Input:
        list_pts_3d: the list of the input points (in 3D)
        jparams:     the parameters of the input for "idw"
    Output:
        (output file written to disk)

    """
    bbox = bounding_box(list_pts_3d)
    power = jparams['power']
    radius1 = jparams['radius1']
    radius2 = jparams['radius2']
    angle = jparams['angle']
    max_points = jparams['max_points']
    min_points = jparams['min_points']
    cellsize, nrows, ncols = cal_col_row_size(list_pts_3d, jparams)
    raster = np.zeros((nrows, ncols))
    points = Point2D(list_pts_3d)
    kd = scipy.spatial.KDTree(points)
    dt = scipy.spatial.Delaunay(points)

    for i in range(nrows):
        for j in range(ncols):
            center = cal_raster_center(bbox, i, j, nrows, cellsize)
            weight = cal_weight(dt, center, kd, radius1, radius2, power, list_pts_3d)
            raster[i][j] = weight

    # print("cellsize:", jparams['cellsize'])

    output_file(raster, jparams['output-file'], list_pts_3d, jparams)
    print("File written to", jparams['output-file'])


def cal_tri_area(pt1, pt2, pt3):
    a = dis(pt1, pt2)
    b = dis(pt1, pt3)
    c = dis(pt2, pt3)
    if a+b<=c or a+c<=b or b + c <= a: return 0
    s = (a+b+c)/2
    area = math.sqrt(s*(s-a)*(s-b)*(s-c))
    return area


def cal_tin_value(dt, center, list_pts_3d):
    pt = []
    z_value = []
    index = scipy.spatial.Delaunay.find_simplex(dt, center)
    if index == -1:
        return nodata_value
    if len(dt.simplices) == 0:
        return nodata_value

    pt.append((list_pts_3d[dt.simplices[index][0]][0], list_pts_3d[dt.simplices[index][0]][1]))
    pt.append((list_pts_3d[dt.simplices[index][1]][0], list_pts_3d[dt.simplices[index][1]][1]))
    pt.append((list_pts_3d[dt.simplices[index][2]][0], list_pts_3d[dt.simplices[index][2]][1]))

    z_value.append(list_pts_3d[dt.simplices[index][0]][2])
    z_value.append(list_pts_3d[dt.simplices[index][1]][2])
    z_value.append(list_pts_3d[dt.simplices[index][2]][2])

    area_0 = cal_tri_area(center, pt[1], pt[2])
    area_1 = cal_tri_area(center, pt[2], pt[0])
    area_2 = cal_tri_area(center, pt[0], pt[1])

    raster_value = (area_0 * z_value[0] + area_1 * z_value[1] + area_2 * z_value[2]) / (area_0 + area_1 + area_2)
    return raster_value


def tin_interpolation(list_pts_3d, jparams):

    bbox = bounding_box(list_pts_3d)
    points = Point2D(list_pts_3d)
    cellsize, nrows, ncols = cal_col_row_size(list_pts_3d, jparams)
    raster = np.zeros((nrows, ncols))
    points = Point2D(list_pts_3d)
    dt = scipy.spatial.Delaunay(points)
    kd = scipy.spatial.KDTree(points)


    for i in range(nrows):
        for j in range(ncols):
            center = cal_raster_center(bbox, i, j, nrows, cellsize)
            raster[i][j] = cal_tin_value(dt, center, list_pts_3d)

    output_file(raster, jparams['output-file'], list_pts_3d, jparams)
    print("File written to", jparams['output-file'])


def create_voronoi(center, list_pts_3d, dt):
    points = Point2D(list_pts_3d)
    index = scipy.spatial.Delaunay.find_simplex(dt, center)
    if index == -1:
        return nodata_value

    neigh=dt.neighbors[index]

    neighbor_coord=[]
    neighbor_distance=[]
    for i in neigh:
        if i == -1:
            continue
        else:
            neighbor_distance.append(dis(center,dt.points[i]))
            neighbor_coord.append(dt.points[i])
    voro_pt=[]
    for item in neighbor_coord:
        voro_pt.append(list(item))
    voro_pt.append(center)
    voro=scipy.spatial.Voronoi(voro_pt)

    #fig=scipy.spatial.voronoi_plot_2d(voro)
    #plt.show()

    ver=voro.vertices
    dict=voro.ridge_dict
    ridge=voro.points

    index=scipy.spatial.Delaunay.find_simplex(dt,center)




def cal_laplace_value(center, ):
    pass


def laplace_interpolation(list_pts_3d, jparams):
    """
    !!! TO BE COMPLETED !!!

    Function that writes the output raster with Laplace interpolation

    Input:
        list_pts_3d: the list of the input points (in 3D)
        jparams:     the parameters of the input for "laplace"
    Output:
        (output file written to disk)

    """
    # -- example to construct the DT with scipy
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html#scipy.spatial.Delaunay
    # dt = scipy.spatial.Delaunay([])

    # -- example to construct the DT with startinpy
    # minimal docs: https://github.com/hugoledoux/startinpy/blob/master/docs/doc.md
    # how to use it: https://github.com/hugoledoux/startinpy#examples
    # you are *not* allowed to use the function for the laplace interpolation that I wrote for startinpy
    # you need to write your own code for this step

    bbox = bounding_box(list_pts_3d)
    points = Point2D(list_pts_3d)
    cellsize, nrows, ncols = cal_col_row_size(list_pts_3d, jparams)
    raster = np.zeros((nrows, ncols))
    points = Point2D(list_pts_3d)
    dt = scipy.spatial.Delaunay(points)
    kd = scipy.spatial.KDTree(points)
    time = 0

    for i in range(nrows):
        for j in range(ncols):
            center = cal_raster_center(bbox, i, j, nrows, cellsize)
            value = create_voronoi(center, list_pts_3d, dt)

    print("File written to", jparams['output-file'])


# -- *all* your code goes into 'my_code_hw01'

def main():
    # -- read the needed parameters from the file 'params.json' (must be in same folder)
    try:
        jparams = json.load(open('params.json'))
    except:
        print("ERROR: something is wrong with the params.json file.")
        sys.exit()
    # -- store the input 3D points in list
    list_pts_3d = []
    with open(jparams['input-file']) as csvfile:
        r = csv.reader(csvfile, delimiter=' ')
        header = next(r)
        for line in r:
            p = list(map(float, line))  # -- convert each str to a float
            assert (len(p) == 3)
            list_pts_3d.append(p)
    # -- interpolations if in the params
    if 'nn' in jparams:
        start_time = time.time()
        print("=== Nearest neighbour interpolation ===")
        nn_interpolation(list_pts_3d, jparams['nn'])
        print("-->%ss" % round(time.time() - start_time, 2))
    if 'idw' in jparams:
        start_time = time.time()
        print("=== IDW interpolation ===")
        idw_interpolation(list_pts_3d, jparams['idw'])
        print("-->%ss" % round(time.time() - start_time, 2))
    if 'tin' in jparams:
        start_time = time.time()
        print("=== TIN interpolation ===")
        tin_interpolation(list_pts_3d, jparams['tin'])
        print("-->%ss" % round(time.time() - start_time, 2))


if __name__ == '__main__':
    main()
