# -*- coding: utf-8 -*-
# Copyright (c) 2017 Interstellar Technologies Inc. All Rights Reserved.
# Authors : Takahiro Inagawa
#
# Lisence : MIT Lisence
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

"""
OpenVerne.IIP - Instantaneous Impact Point(IIP) calculation

The instantaneous impact point (IIP) of a rocket, given its position and velocity,
is defined as its touchdown point(altitude=0[m]) assuming a free-fall flight (without propulsion).
The IIP is considered as a very important information for safe launch operation
of a rocket.

cf.
Jaemyung Ahn and Woong-Rae Roh.  "Noniterative Instantaneous Impact Point Prediction Algorithm for Launch Operations",
Journal of Guidance, Control, and Dynamics, Vol. 35, No. 2 (2012), pp. 645-648.
https://doi.org/10.2514/1.56395
"""

import numpy as np
from numpy import cos, sin, tan, arcsin, arctan2, arccos
from numpy import sqrt, deg2rad, rad2deg, pi

class IipConstants():
    def __init__(self):
        self.re_a = 6378137.0  # Long axis of WGS84 in meters
        self.e1 = 8.1819190842622e-2  # First Eccentricity
        self.one_f = 298.257223563  # 1/f of flatness f (smoothness)
        self.re_b = 6356752.314245 # Short axis of WGS84 in meters
        self.e2 = 6.6943799901414e-3  # square of first eccentricity e
        self.ed2 = 6.739496742276486e-3  # square of the second eccentricity e'

wgs84 = IipConstants()


def posECEF_from_LLH(posLLH_):
    """
    Args:
        posLLH_ (np.array 3x1) : position in LLH coordinate [deg, deg, m]
    Return:
        (np.array 3x1) : position in ECEF coordinate [m, m, m]
    """
    lat = deg2rad(posLLH_[0])
    lon = deg2rad(posLLH_[1])
    alt = posLLH_[2]
    W = sqrt(1.0 - wgs84.e2 * sin(lat) * sin(lat))
    N = wgs84.re_a / W
    pos0 = (N + alt) * cos(lat) * cos(lon)
    pos1 = (N + alt) * cos(lat) * sin(lon)
    pos2 = (N * (1 - wgs84.e2) + alt) * sin(lat)
    return np.array([pos0, pos1, pos2])

def dcmECI2ECEF(second, omega):
    """
    Args:
        second (double) : time from reference time[s]
    Return:
        dcm (np.array 3x3) : DCM from ECI to ECEF
    """
    theta = omega * second
    dcm = np.array([[cos(theta),  sin(theta), 0.0],
                    [-sin(theta), cos(theta), 0.0],
                    [0.0,         0.0,        1.0]])
    return dcm

def posECI(posECEF_, second, omega):
    """
    Args:
        posECEF_ (np.array 3x1) : position in ECEF coordinate [m, m, m]
        second (double) : time from reference time [s]
    Return:
        (np.array 3x1) : position in ECI coordinate [m, m, m]
    """
    dcmECI2ECEF_ = dcmECI2ECEF(second, omega)
    dcmECEF2ECI_ = dcmECI2ECEF_.T
    return dcmECEF2ECI_.dot(posECEF_)

def dcmECEF2NED(posLLH_):
    """
    Args:
        posLLH_ (np.array 3x1) : [deg, deg, m]
    Return:
        dcm (np.array 3x3) : DCM from ECEF to NED
    """
    lat = deg2rad(posLLH_[0])
    lon = deg2rad(posLLH_[1])
    dcm = np.array([[-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)],
                    [-sin(lon),           cos(lon),          0],
                    [-cos(lat)*cos(lon), -cos(lat)*sin(lon), -sin(lat)]])
    return dcm

def dcmECI2NED(dcmECEF2NED_, dcmECI2ECEF_):
    return dcmECEF2NED_.dot(dcmECI2ECEF_)


def posLLH(posECEF_):
    """
    Args:
        posECEF_ (np.array 3x1) : [deg, deg, m]
    Return:
        (np.array 3x1) : position in LLH coordinate [deg, deg, m]
    """
    def n_posECEF2LLH(phi_n_deg):
        return wgs84.re_a / sqrt(1.0 - wgs84.e2 * sin(deg2rad(phi_n_deg)) * sin(deg2rad(phi_n_deg)))
    p = sqrt(posECEF_[0] **2 + posECEF_[1] **2)
    theta = arctan2(posECEF_[2] * wgs84.re_a, p * wgs84.re_b) # rad
    lat = rad2deg(arctan2(posECEF_[2] + wgs84.ed2 * wgs84.re_b * pow(sin(theta), 3), p - wgs84.e2 * wgs84.re_a * pow(cos(theta),3)))
    lon = rad2deg(arctan2(posECEF_[1], posECEF_[0]))
    alt = p / cos(deg2rad(lat)) - n_posECEF2LLH(lat)
    return np.array([lat, lon, alt])


def lat_from_radius(radius):
    """ Return latitude[deg] from earth radius
    Args:
        radius (double) : earth radius[m]
    Return:
        (double) : latitude [deg]
    """
    lat_rad =  arcsin(sqrt(1/wgs84.e2 * (1 - (radius**2) / (wgs84.re_a**2))))
    return rad2deg(lat_rad)

class IIP:
    def __init__(self):
        self.mu = 3.986004418 * 10**(14)  # Earth's gravitational constant m3s-2
        self.omega_earth = 7.2921159e-5; # rotation angular velocity of the earth [rad/s]
        self.epsilon = 1e-3  # convergence error for convergence calculation
        return
        
    def compute_iip(self, vehicle_position_llh, vehicle_velocity_ned):
        """ calculate IIP from current position(LLH) & current velocity(NED)
        Args:
            posLLH_ (np.array 3x1) : position at current point(LLH) [deg, deg, m]
            velNED_ (np.array 3x1) : velocity at current point(NED frame) [m/s, m/s, m/s]
        Attributes:
            posLLH_IIP_deg (np.array 2x1) : IIP position (LLH coordinate) [deg, deg]
            posLLH_IIP_rad (np.array 2x1) : IIP position (LLH coordinate) [rad, rad]
            distance_ECEF (double) : earth surface distance from current point to IIP [m]
            tf (double) : time of flight from current point to IIP [s]
        Usage:
            > _IIP = IIP(posLLH_, velNED_)
            > print(_IIP)
        """

        # Conversion of initial position/velocity to ECI coordinate system
        posLLH_ = vehicle_position_llh
        velNED_ = vehicle_velocity_ned
        posECI_init_ = posECI(posECEF_from_LLH(posLLH_), 0, self.omega_earth)
        dcmECI2NED_ = dcmECI2NED(dcmECEF2NED(posLLH_), dcmECI2ECEF(0, self.omega_earth))
        omegaECI2ECEF_ = np.array([[0.0,         -self.omega_earth, 0.0],
                                   [self.omega_earth, 0.0,          0.0],
                                   [0.0,         0.0,          0.0]])  # angular velocity tensor
        velECI_init_ = np.dot(dcmECI2NED_.transpose(), velNED_) + omegaECI2ECEF_.dot(posECI_init_)

        # Absolute values ​​and unit vectors of r0 and v0 required for calculation, calculation of initial γ:flight-path angle
        r0 = np.linalg.norm(posECI_init_)
        v0 = np.linalg.norm(velECI_init_)
        ir0 = posECI_init_ / np.linalg.norm(posECI_init_)  # unit vector of positions
        iv0 = velECI_init_ / np.linalg.norm(velECI_init_)  # unit vector of velocity
        gamma0 = arcsin(np.dot(ir0, iv0))  # [rad]
        #self.gamma0 = gamma0  # to an instance variable to see the gamma from the outside

        lam = v0 ** 2 / (self.mu / r0)  # lambda

        rp1 = wgs84.re_b  # Lower Interval of Bisection for Convergence Calculations Minor Axis of the Earth
        rp2 = wgs84.re_a  # Upper section Earth's semimajor axis

        phi = 0
        ip = 0
        while True:  # dichotomy

            rpM = (rp1 + rp2) / 2

            # Calculation of phi: flight angle
            phi = self.compute_vehicle_flight_angle(gamma0, lam, r0, rpM)
            ip = self.compute_ip(gamma0, phi, ir0, iv0)

            #hantei_rp1 = rp_calc(rp1)  # positive or negative nan
            #hantei_rp2 = rp_calc(rp2)  # positive or negative nan

            hantei_rpM = self.compute_rp_magnitude(ip, rpM)

            #print("rpM = %.1f, 1:%.5f, 2:%.5f, M:%.5f" % (rpM, hantei_rp1, hantei_rp2, hantei_rpM))

            if (hantei_rpM < 0):

                rp1 = rpM

            else:

                rp2 = rpM

            if (abs(hantei_rpM) < self.epsilon):

                break

        # set converged rp value as rp
        rp = rpM

        IIP_LLH_deg = posLLH(ip * rp)
        lat_ECI_IIP_rad = deg2rad(IIP_LLH_deg[0])
        lon_ECI_IIP_rad = deg2rad(IIP_LLH_deg[1])

        posLLH_ECI_IIP_rad = np.array([lat_ECI_IIP_rad, lon_ECI_IIP_rad, 0])
        posLLH_init_rad = np.zeros(3)
        posLLH_init_rad[0] = deg2rad(posLLH_[0])
        posLLH_init_rad[1] = deg2rad(posLLH_[1])

        # Earth surface distance from initial position to IIP
        # cf. https://keisan.casio.jp/exec/system/1257670779
        self.distance_ECI = wgs84.re_a * arccos(sin(posLLH_init_rad[0])*sin(posLLH_ECI_IIP_rad[0]) + cos(posLLH_init_rad[0])*cos(posLLH_ECI_IIP_rad[0])*cos(posLLH_ECI_IIP_rad[1]-posLLH_init_rad[1]))

        # Flight time calculation Reference: eq.(19)
        self.tf = self.compute_flight_time(r0, v0, gamma0, phi, lam)

        # Calculate the latitude and longitude of the landing position from the flight time, considering the rotation of the earth Reference: eq(14),(15)
        lat_ECEF_IIP_rad = lat_ECI_IIP_rad
        lon_ECEF_IIP_rad = lon_ECI_IIP_rad - self.omega_earth * self.tf
        self.posLLH_IIP_rad = np.array([lat_ECEF_IIP_rad, lon_ECEF_IIP_rad])
        self.posLLH_IIP_deg = np.array([rad2deg(lat_ECEF_IIP_rad), rad2deg(lon_ECEF_IIP_rad)])

        # Earth surface distance from initial position to IIP
        self.distance_ECEF = wgs84.re_a * arccos(sin(posLLH_init_rad[0])*sin(self.posLLH_IIP_rad[0]) + cos(posLLH_init_rad[0])*cos(self.posLLH_IIP_rad[0])*cos(self.posLLH_IIP_rad[1]-posLLH_init_rad[1]))
    
        return 
    
    def iip_result_lat(self):

        return self.posLLH_IIP_deg[0]
    
    def iip_result_lon(self):

        return self.posLLH_IIP_deg[1]

    def compute_vehicle_flight_angle(self, gamma0, lam, pos0, rp):

        c1 = - tan( gamma0 )
        c2 = 1 - 1 / ( lam * cos( gamma0 ) ** 2 )
        c3 = pos0 / rp - 1 / ( lam * cos( gamma0 ) ** 2 )
        c12 = c1 ** 2
        c22 = c2 ** 2
        c32 = c3 ** 2

        try:
            phi = arcsin( ( c1 * c3 + sqrt( c12 * c32 - ( c12 + c22 ) * ( c32 - c22 ) ) ) / (c12 + c22) )
        except RuntimeWarning:
            phi = np.nan

        return phi

    def compute_rp_magnitude(self, ip, radius):

        # Unit vector of IIP position and IIP latitude and longitude in ECI coordinate system calculated from it Reference: eq.(13)~(15)
        IIP_LLH_deg = posLLH(ip * radius)
        lat_ECI_IIP_rad = deg2rad(IIP_LLH_deg[0])
        #lon_ECI_IIP_rad = deg2rad(IIP_LLH_deg[1])

        #print("phi = %3f [deg], lat IIP %.3f [deg]" % (rad2deg(phi),rad2deg(lat_ECI_IIP_rad)))
        rp_new = wgs84.re_a * sqrt(1 - wgs84.e2 * sin(lat_ECI_IIP_rad)**2)

        return radius - rp_new

    def compute_ip(self, gamma0, phi, ir0, iv0):

        ip = cos( gamma0 + phi ) / cos( gamma0 ) * ir0 + sin( phi ) / cos( gamma0 ) * iv0  # IIP Unit Vector (ECI)
        ip = ip / np.linalg.norm(ip)

        return ip

    def compute_flight_time(self, r0, v0, gamma0, phi, lam):

        t1 = r0 / v0 / cos( gamma0 )
        t2 = tan( gamma0 ) * ( 1 - cos( phi ) ) + ( 1 - lam ) * sin( phi )
        t3 = ( 2 - lam ) * ( ( 1 - cos( phi  ) ) / ( lam * cos( gamma0 ) ** 2 ) )
        t4 = ( 2 - lam ) * ( cos( gamma0 + phi ) / cos( gamma0 ) )
        t5 = 2 * cos( gamma0 ) / ( lam * ( 2 / lam - 1 ) ** 1.5 )
        t6u = sqrt( 2 / lam - 1 )
        t6l = cos( gamma0 ) * tan( pi/2 - phi / 2 ) - sin( gamma0 )

        return t1 * ( ( t2 / ( t3 + t4 ) ) + t5 * arctan2( t6u, t6l ) )
    
    def __repr__(self):
        print("==== current point ====")
        print("lat = %.6f [deg], lon = %.6f [deg]" % (self.posLLH_[0], self.posLLH_[1]))
        print("altitude = %.1f [m]" %(self.posLLH_[2]))
        print("velocity(NED) = %.1f [m/s], %.1f [m/s], %.1f [m/s]" % (self.velNED_[0], self.velNED_[1], self.velNED_[2]))
        print("r0(ECI) = %.1f [m]" % (self.r0))
        print("v0(ECI) = %.1f [m/s]" % (self.v0))
        print("unit vector of r0 (ECI) = [%.6f, %.6f, %.6f]" % (self.ir0[0], self.ir0[1], self.ir0[2]))
        print("unit vector of v0 (ECI) = [%.6f, %.6f, %.6f]" % (self.iv0[0], self.iv0[1], self.iv0[2]))
        print("gamma0 = %.4f [deg]" % (rad2deg(self.gamma0)))
        print("==== IIP (Instantaneous Impact Point) ====")
        print("lat = %.6f [deg], lon = %.6f [deg]" % (self.posLLH_IIP_deg[0], self.posLLH_IIP_deg[1]))
        print("distance of earth surface ECEF = %.1f [m]" % (self.distance_ECEF))
        print("time of flight = %.2f [s]" % (self.tf))
        print("distance of earth surface ECI  = %.1f [m]" % (self.distance_ECI))
        print("flight angle of a rocket = %.6f [deg]" % (rad2deg(self.phi)))
        print("unit vector of IIP(ECI) = [%.6f, %.6f, %.6f]" % (self.ip[0], self.ip[1], self.ip[2]))
        return ""

    def disp(self):
        """ Simple display of result """
        print("==== current point ====")
        print("lat = %.6f [deg], lon = %.6f [deg]" % (self.posLLH_[0], self.posLLH_[1]))
        print("altitude = %.1f [m]" %(self.posLLH_[2]))
        print("velocity(NED) = %.1f [m/s], %.1f [m/s], %.1f [m/s]" % (self.velNED_[0], self.velNED_[1], self.velNED_[2]))
        print("==== IIP (Instantaneous Impact Point) ====")
        print("lat = %.6f [deg], lon = %.6f [deg]" % (self.posLLH_IIP_deg[0], self.posLLH_IIP_deg[1]))
        print("distance of earth surface = %.1f [m]" % (self.distance_ECEF))
        print("time of flight = %.2f [s]" % (self.tf))

if __name__ == '__main__':

    posLLH_ = np.array([40, 140, 100])
    velNED_ = np.array([10, 0, 0])

    _IIP = IIP(posLLH_, velNED_)
    print(_IIP)
