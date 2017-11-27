# Molecular Dynamics: The simulation of a collision between two Lennard-Jones clusters.
A simple Molecular Dynamics code written from scratch for a Computational Laboratory assignment

A simple molecular dynamics program simulating the collision of two Lennard-Jones
clusters, written in Python 3. The program computes the Kinetic Energy, Force
and Total Energy as function of time for the Lennard-Jones system. The structure of the cluster
after collison is printed to an output file of the .xyz format. For the integration
of Newton's equations of motion the verlet algorithm is used, and the
Lennard-Jones potential is truncated using 'Simple Truncation' for simplicity.

To run the code simply cd to the location it is saved and type "python LJCollision.py".
The program then requests the time-step, number of steps, initial separation and inital
velocity as input. The rest is automated.

The code is well annotated, so feel free to dig around if you so desire.

Callum Hutchinson, King's College London, December 2016.

Contact:
callum.hutchinson@kcl.ac.uk
