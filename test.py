# -*- coding: utf-8 -*-
""" Simplest example """

from OpenVerne import IIP
import matplotlib.pyplot as plt
import csv
import numpy as np

iip_lons = []
iip_lats = []

#
# Test to ensure that logic changes produce the same result
#
with open('./testing/data.csv') as csv_file:

    reader = csv.DictReader(csv_file)
    iip = IIP()
    for row in reader:

        lat = float(row['lat'])
        lon = float(row['lon'])
        alt = float(row['alt'])
        n_vel = float(row['n_vel'])
        e_vel = float(row['e_vel'])
        d_vel = float(row['d_vel'])
        iip_recorded_lon = float(row['iip_lon'])
        iip_recorded_lat = float(row['iip_lat'])

        pos_llh = np.array([lat, lon, alt])
        vel_ned = np.array([n_vel, e_vel, d_vel])
        iip.compute_iip(pos_llh, vel_ned)
        iip_x = iip.iip_result_lon()
        iip_y = iip.iip_result_lat()
        
        print('iip_x: %f\t\tiip_actual: %f' % (iip_x, iip_recorded_lon))
        assert np.isclose(iip_x, iip_recorded_lon, rtol=1e-08, atol=1e-08, equal_nan=False)

        print('iip_y: %f\t\tiip_actual: %f' % (iip_y, iip_recorded_lat))
        assert np.isclose(iip_y, iip_recorded_lat, rtol=1e-08, atol=1e-08, equal_nan=False)