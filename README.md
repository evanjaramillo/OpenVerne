# OpenVerne - Instantaneous Impact Point (IIP) Calculation

### This is a fork of the original repository that can be found [here](https://github.com/istellartech/OpenVerne).

The instantaneous impact point (IIP) of a rocket, given its position and velocity,
is defined as its touchdown point assuming a free-fall flight without propulsion.
The IIP is a very important metric to monitor during launch operations to ensure a safe flight.

A script that can calculate the instantaneous impact point (IIP) of a rocket
Enter latitude [deg] longitude [deg] altitude [m], north speed [m/s] east speed [m/s] and vertical downward speed [m/s]
Coordinates of IIP are output.

## Usage
see example(example_xx.py in this repository).

## Coordinate
Regarding the position, it must be inputted at latitude and longitude altitude.(LLH)

For speed, input it in the North, East and Down direction coordinate system at that latitude, longitude and altitude.(NED)

## IIP class constractor

```python
IIP(np.array([latitude, longitude, altitude]), np.array([North, East, Down]))
```

The first argument is the position in the LLH coordinate system [deg, deg, m]

The second argument is the speed in the NED coordinate system [m/s, m/s, m/s]

## References
Jaemyung Ahn and Woong-Rae Roh.  "Noniterative Instantaneous Impact Point Prediction Algorithm for Launch Operations",
Journal of Guidance, Control, and Dynamics, Vol. 35, No. 2 (2012), pp. 645-648.
https://doi.org/10.2514/1.56395

Young-Woo Nam, Taehyun Seong, and Jaemyung Ahn.  "Adjusted Instantaneous Impact Point and New Flight Safety Decision Rule", Journal of Spacecraft and Rockets, Vol. 53, No. 4 (2016), pp. 766-773.
https://doi.org/10.2514/1.A33424

## Conception of name
[Jules Verne](https://en.wikipedia.org/wiki/Jules_Verne) ([From_the_Earth_to_the_Moon](https://en.wikipedia.org/wiki/From_the_Earth_to_the_Moon))


## License
OpenVerne is an Open Source project licensed under the MIT License
