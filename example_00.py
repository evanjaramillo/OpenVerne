# -*- coding: utf-8 -*-
""" Simplest example """

from OpenVerne import IIP
import matplotlib.pyplot as plt
import csv
import numpy as np

with open('data.csv') as csv_file:
    reader = csv.DictReader(csv_file)
    i = 0
    for row in reader:
        lat = float(row['lat'])
        lon = float(row['lon'])
        alt = float(row['alt'])
        n_vel = float(row['n_vel'])
        e_vel = float(row['e_vel'])
        d_vel = float(row['d_vel'])
        pos_llh = np.array([lat, lon, alt])
        vel_ned = np.array([n_vel, e_vel, d_vel])
        iip = IIP(pos_llh, vel_ned)
        iip_lat = iip.iip_result_lat()
        iip_lon = iip.iip_result_lon()

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        plt.xlabel('Longitude (deg)')
        plt.ylabel('Latitude (deg)')
        plt.title('IIP vs Position Over Time')
        plt.plot(lon, lat, 'bo')
        plt.plot(iip_lon, iip_lat, 'rx')

        plt.savefig('iip_out_' + str(i) + '.png')
        plt.close()
        i = i + 1
