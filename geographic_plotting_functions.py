#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 09:44:25 2021

@author: rntroyer
"""

import numpy as np
from scipy.interpolate import interp1d

def rad(degree):
    """Function to convert from degree to radians
    DEPENDENCIES
        numpy
    INPUT
    degree
        type: float
        about: angle in degrees
    OUTPUT
        type: float
        about: angle in radians
    """
    return degree*(np.pi/180)

def deg(radian):
    """Function to convert from radians to degrees
    DEPENDENCIES
        numpy
    INPUT
    radian
        type: float
        about: angle in radians
    OUTPUT
        type: float
        about: angle in degrees
    """
    return radian*(180/np.pi)


def quadratic_solver(a, b, c):
    """Function to solve the quadratic equation with real solutions
    of the form a*x^2 + b*x + c. If there are no real solutions
    function does not return a value.
    DEPENDENCIES
        numpy
    INPUT
    a, b, c
        type: float
        about: equation coefficients
    OUTPUT
        type: list
        about: the two different quadratic equation solutions
    """
    if b**2-(4*a*c)<0:
        print('Quadratic equation had no real solutions.')
    else:
        return [(-b+np.sqrt(b**2-(4*a*c)))/(2*a),\
                (-b-np.sqrt(b**2-(4*a*c)))/(2*a)]

def ground_projection(angle=0, edge_buffer=0,
                      altitude=100, radius=6371,
                      ):
    """Function to calculate the ground distance projection of a
    line up into the atmosphere.
    DEPENDENCIES
        numpy
    INPUT
    angle = 0
        type: float
        about: elevation angle with the ground in radians
    edge_buffer=0
        type: float
        about: buffer angle, so edge of fov isn't used if desired
    altitude = 100
        type: float
        about: height in kilometers in the atmosphere that the line goes,
                set to approximate auroral height
    radius = 6371
        type: float
        about: radius in kilometers of the sphere,
              set to the earth radius
    OUTPUT
    projection
        type: float
        about: distance in kilometers of projection on the ground
    """
    # Calculate law of cosines
    gamma = rad(90) + angle
    B = radius
    C = radius + altitude

    a = 1
    b = -2*B*np.cos(gamma)
    c = B**2-C**2

    # Solve quadratic equation
    A = max(quadratic_solver(a, b, c))

    # Check answer is reasonable
    if A<=0:
        print('Result from quadratic equation is unreasonable.'
              ' Stopping.', A)
        exit()

    # Solve for field of view angle from law of cosines
    alpha = np.arccos((A**2-B**2-C**2)/(-2*B*C))

    # Spacial size of field of view (arc length)
    projection = alpha*radius - edge_buffer

    return projection

def direct_geodetic_problem_calc(start_point, bearing,
                                 angular_distance,
                                 N = np.array([0, 0, 1])):
    """Function to solve the direct geodetic problem. This calculates
    the latitude and longitude of a point given a starting point,
    bearing to the new point, and distance to the new point.
    For more info:
    https://www.movable-type.co.uk/scripts/latlong-vectors.html
    DEPENDENCIES
        numpy
    INPUT
    start_point
        type: array
        about: cartesian vector normal to the surface
                representing the starting point.
    bearing
        type: float
        about: angle in radians, clockwise from north, from the
               starting point to the new point.
    angular_distance
        type: float
        about: angular distance between the starting point and new point
    N
        type: array
        about: array representing the north pole.
    OUTPUT
    lat
        type: float
        about: latitude of the new point
    lon
        type: float
        about: longitude of the new point
    """
    # East vector at point a
    d_e = np.cross(N, start_point)

    # North vector at a
    d_n = np.cross(start_point, d_e)

    # Normalized vector in direction of theta
    d = d_n*np.cos(bearing) + d_e*np.sin(bearing)
    b = (start_point*np.cos(angular_distance)
         + d*np.sin(angular_distance))

    # Convert to latitude and longitude
    lat = np.arctan2(b[2], np.sqrt(b[0]**2 + b[1]**2))
    lon = np.arctan2(b[1], b[0])

    return lon, lat

def asi_pixel_map(image_width, image_height, asi_lon, asi_lat,
                  fov_radius, x_shift=0, y_shift=0,
                  earth_radius = 6371,
                  return_all = False):
    """Function to calculate meshgrids for the allsky camera image
       with latitude and longitude values for each pixel.
    DEPENDENCIES
        numpy, direct_geodetic_problem_calc, ground_projection
    INPUT
    image_width, image_height
        type: int
        about: width and height of the image in pixels
    asi_lat, asi_long
        type: float
        about: latitude and longitude of asi camera
    fov_radius
        type: int
        about: radius in pixels of the fov
    x_shift=0, y_shift=0
        type: int
        about: x and y difference in pixels between
              fov center and image center
    earth_radius
        type: int
        about: radius of the earth in kilometers
    return_all = False
        type: bool
        about: whether to return more than just pixel_lon and pixel_lat
    OUTPUT
    pixel_lon, pixel_lat
        type: array
        about: grid arrays with longitudes and latitudes of each pixel
    polar_angle_grid
        type: array
        about: array with polar angle of each pixel
               only applies if return_all = True
    horizon_angle_grid
        type: array
        about: array with angle from horizon of each pixel
                only applies if return_all = True
    """

    # Construct a grid of x and y locations centered around image middle
    x = np.linspace(-image_width/2, image_width/2,
                    image_width) - x_shift
    y = np.linspace(-image_height/2, image_height/2,
                    image_height) - y_shift
    x_grid, y_grid = np.meshgrid(x, y)

    # Get the radial distance of each pixel
    polar_radius_grid = np.sqrt(x_grid**2 + y_grid**2)

    # Get the angle from north vector of each pixel
    #...this has to be done for all 4 quadrants

    # Quadrant 1
    #...x values positive, y values positive
    #...create array for this quadrant, but will fix other quadrants
    polar_angle_grid = rad(90) - np.arctan(np.abs(y_grid/x_grid))

    # Quadrant 2
    #...x values positive, y negative
    cond_q2 = (x_grid > 0) & (y_grid < 0)
    polar_angle_grid[cond_q2] = (rad(180) - polar_angle_grid[cond_q2])

    # Quadrant 3
    #...x negative, y negative
    cond_q3 = (x_grid < 0) & (y_grid < 0)
    polar_angle_grid[cond_q3] = (rad(180) + polar_angle_grid[cond_q3])

    # Quadrant 4
    #...x negative, y positive
    cond_q4 = (x_grid < 0) & (y_grid > 0)
    polar_angle_grid[cond_q4] = (rad(360) - polar_angle_grid[cond_q4])

    # Get angle from horizon to point in sky
    horizon_angle_grid = rad(90 - (polar_radius_grid/fov_radius)*90)

    # Get the as-the-crow-flies distance to under aurora
    projected_distance = [[ground_projection(angle=y) for y in x]
                          for x in horizon_angle_grid]

    # Convert to array
    projected_distance = np.array(projected_distance)

    # Get these values as an angular distance
    angular_projected_distance = projected_distance/earth_radius

    # Loop through each pixel and calculate its latitude and longitude
    #...this will also calculate it for borders around camera FOV
    #...these aren't physical and so can be masked out later
    pixel_lat = np.zeros([image_height, image_width])
    pixel_lon = np.zeros([image_height, image_width])

    # Calculate vector defining location of ASI
    asi_vec = [np.cos(rad(asi_lat))*np.cos(rad(asi_lon)),
               np.cos(rad(asi_lat))*np.sin(rad(asi_lon)),
               np.sin(rad(asi_lat))]
    asi_vec = np.array(asi_vec)

    for w in range(0, image_width):

        for h in range(0, image_height):

            # Get latitude and longitude
            lon, lat = direct_geodetic_problem_calc(asi_vec,
                                    polar_angle_grid[h, w],
                                    angular_projected_distance[h, w])

            # Write values to array, converting back to degree
            pixel_lat[h, w] = deg(lat)
            pixel_lon[h, w] = deg(lon)

    if return_all == True:
        return (pixel_lon, pixel_lat,
                polar_angle_grid, horizon_angle_grid,
                x_grid, y_grid)

    else:
        return pixel_lon, pixel_lat

def elevation_angle_from_distance(distance, altitude=100):
    """Function to get the angle above the horizon for a specified
    distance (in km) away beneath the specified altitude.
    DEPENDENCIES
        numpy, scipy.interpolate.interp1d, ground_projection
    INPUT
    distance
        type: float
        about: distance in kilometers from asi location
    altitude=100
        type: float
        about: height in atmosphere that distance is projected up to
    OUTPUT
    model(distance)
        type: 0D array
        about: modeled angle above horizon that matches with distance.
    """

    # Construct points to use for model

    # Radian angles between 0 (horizon) and 90 degrees (directly above)
    angles = np.linspace(0, np.pi/2, 1000)

    # Distances calculated with the ground projection function
    distances = [ground_projection(a, altitude=altitude) for a in angles]

    # Construct an interpolated model from these points
    model =  interp1d(distances, angles)

    return model(distance)

def n_vector(lon, lat):
    """Function to create a 3D cartesian vector defining
    latitude and longitude point on surface of sphere, where
    north pole = [0, 0, 1].
    DEPENDENCIES
        numpy
    INPUT
    lon, lat
        type: float
        about: longitude and latitude values of point
    OUTPUT
    n_vec
        type: array
        about: 3D vector defining point in cartesian space.
    """

    n_vec = [np.cos(rad(lat))*np.cos(rad(lon)),
             np.cos(rad(lat))*np.sin(rad(lon)),
             np.sin(rad(lat))]
    n_vec = np.array(n_vec)

    return n_vec

def distance_to(asi_vec, point_vec, radius=6371):
    """Function to calculate the distance between two points
    on the surface of a sphere. For more info:
    https://www.movable-type.co.uk/scripts/latlong-vectors.html
    DEPENDENCIES
        numpy
    INPUT
    asi_vec
        type: array
        about: 3D cartesian vector defining starting point.
    point_vec
        type: array
        about: 3D cartesian vector defining ending point.
    radius=6371
        type: float
        about: radius in kilometers of the sphere. Set to earth radius
    OUTPUT
        type: float
        about: distance in kilometers between starting and ending points
    """

    input1 = np.linalg.norm(np.cross(asi_vec, point_vec))
    input2 = np.dot(asi_vec, point_vec)

    return radius*np.arctan2(input1, input2)

def bearing_to(asi_vec, point_vec):
    """Function to calculate the bearing between a great circle
    connecting the north pole and an asi location and the great
    circle connecting the asi location and a different point.
    More info:
    https://www.movable-type.co.uk/scripts/latlong-vectors.html
    DEPENDENCIES
        numpy
    INPUT
    asi_vec
        type: array
        about: 3D cartesian vector defining starting point.
    point_vec
        type: array
        about: 3D cartesian vector defining ending point.
    OUTPUT
    theta
        type: float
        about: angle in radians
    """

    # Calculate bearing between asi location and point
    N = np.array([0, 0, 1])

    # Calculate vectors representing great circles
    c1 = np.cross(asi_vec, point_vec)
    c2 = np.cross(asi_vec, N)

    # Calculate sin of theta
    angle_sign = np.sign(np.dot(np.cross(c1, c2), asi_vec))
    sin_theta = np.linalg.norm(np.cross(c1, c2))*angle_sign

    # Calculate cos of theta
    cos_theta = np.dot(c1, c2)

    # Finally get bearing
    theta = np.arctan2(sin_theta, cos_theta)

    return theta
