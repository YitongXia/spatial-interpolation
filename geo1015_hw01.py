# -- geo1015_hw01.py
# -- hw01 GEO1015.2021
# -- Hugo Ledoux <h.ledoux@tudelft.nl>
# -- 2021-11-10

# ------------------------------------------------------------------------------
# DO NOT MODIFY THIS FILE!!!
# ------------------------------------------------------------------------------

import sys
import math
import csv
import random
import json
import time

# -- *all* your code goes into 'my_code_hw01'
import my_code_hw01


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
        my_code_hw01.nn_interpolation(list_pts_3d, jparams['nn'])
        print("-->%ss" % round(time.time() - start_time, 2))
    if 'idw' in jparams:
        start_time = time.time()
        print("=== IDW interpolation ===")
        my_code_hw01.idw_interpolation(list_pts_3d, jparams['idw'])
        print("-->%ss" % round(time.time() - start_time, 2))
    if 'tin' in jparams:
        start_time = time.time()
        print("=== TIN interpolation ===")
        my_code_hw01.tin_interpolation(list_pts_3d, jparams['tin'])
        print("-->%ss" % round(time.time() - start_time, 2))
    if 'laplace' in jparams:
        start_time = time.time()
        print("=== Laplace interpolation ===")
        my_code_hw01.laplace_interpolation(list_pts_3d, jparams['laplace'])
        print("-->%ss" % round(time.time() - start_time, 2))


if __name__ == '__main__':
    main()
