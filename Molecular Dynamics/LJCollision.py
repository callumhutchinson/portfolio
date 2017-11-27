#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~Lennard-Jones Cluster Collision Simulation~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#A simple molecular dynamics program simulating the collision of two Lennard-Jones
#clusters, written in Python 3.5.2. The program computes the Kinetic Energy, Force
#and Total Energy as function of time for the Lennard-Jones system. The structure of the cluster
#after collison is printed to an output file of the .xyz format. For the integration
#of Newton's equations of motion the verlet algorithm is used, and the
#Lennard-Jones potential is truncated using 'Simple Truncation' for simplicity.

#Callum Hutchinson, King's College London, December 2016.
#
#Contact:
#callum.hutchinson@kcl.ac.uk
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#CODE STARTS HERE#

#import some relevant python libraries to aid our calculations and plotting
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os

# A function which loads cluster co-ordinates into a matrix from two external files
def get_clusters(filename1,filename2):
	cluster1 = pd.read_csv(filename1,header=None,delim_whitespace=True).as_matrix()
	cluster2 = pd.read_csv(filename2,header=None,delim_whitespace=True).as_matrix()
	return cluster1,cluster2

#function which initialises the positions, velocities, forces and energies 
#of the two clusters
def init(cluster1,cluster2,init_dist,init_vel):
	row1,col1 = cluster1.shape
	row2,col2 = cluster2.shape
	Npart = row1+row2

	#initialises the clusters with cluster 1 centered at the origin by default
	#cluster two is set at some initial distance along the x-axis.
	#then combines each cluster into one matrix.
	c1_pos =cluster1
	cluster2[:,0] =cluster2[:,0] + init_dist
	c2_pos = cluster2
	c_pos = np.vstack((c1_pos,c2_pos))
	
	#initialises cluster 1 at the origin to be stationary, and cluster two to
	#be moving towards it at some initial velocity along the x-axis.
	vel1 = np.zeros([row1,col1])
	vel2 = np.zeros([row2,col2])
	vel2[:,0] = vel2[:,0] - init_vel
	c_vel = np.vstack((vel1,vel2))

	#initialises the force at zero
	frc = np.zeros([Npart,col1])

	#initialises the potential energy of the system to zero
	en_pot = np.zeros([Npart,1])
	en_pot = en_pot.reshape(Npart,1)

	#initialises the kinetic energy
	en_kin = abs(c_vel).sum(axis=1)
	en_kin = ((en_kin.reshape(Npart,1)**2)*0.5)

	return c_pos,c_vel,frc,en_pot,en_kin,Npart,init_dist


def get_force(c_pos,frc,en_pot,Npart):
	#forces and potential energies initialsied to zero again at
	#start of each get_force run.
	en_pot = np.zeros([Npart,1])
	en_pot = en_pot.reshape(Npart,1)
	frc = np.zeros([Npart,3])

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#~~~~~~~~~~~~~~~~~~~~~VARIABLE BLOCK~~~~~~~~~~~~~~~~~~~~~~#
	
	#These codes become awfully cluttered after a while, and
	#although Python doesn't use the convention of variable
	#declaration blocks like FORTRAN, it is convenient to use
	#them in times like these for the purposes of 
	#organisation/sanity.

	#sigma value ~1.122 sigma AA (reduced units)
	sigma = 1.0

	#epsilon value 0.010394 eV (reduced units)
	eps = 1.0

	#cut-off radius, cut-off distance = 2.5*sigma=3.82AA
	rcut = 2.5

	#cut-off radius squared
	rc2 = rcut**2

	#cut-off potential energy (Frenkel-Smit)
	ecut = 4*(1/(rcut**12)-1/(rcut**6))


	#~~~~~~~~~~~~~~~~~~END OF VARIABLE BLOCK~~~~~~~~~~~~~~~~#
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

	#calculating the distance between each pair in the system
	for i in range(Npart-1):
		for j in range(i+1,Npart):
			xr = c_pos[i,0] - c_pos[j,0]
			yr = c_pos[i,1] - c_pos[j,1]
			zr = c_pos[i,2] - c_pos[j,2]
			r_sqr = xr**2 + yr**2 + zr**2

			#if within cutoff limit, calculate force using LJ potential (force is deriv. of pot)
			if r_sqr < rc2:
				r2i = 1/r_sqr
				r6i = r2i**3
				#calculate the lennard jones potential Fx=dU/dx=dU/dr*dr/dx=dU/dr*(x/r)
				ff = 48*r2i*(r6i**2-0.5*r6i) #this is the dU/dx term, times by the distance to calc force
				#update the forces, negative for one because Fij = - Fji
				frc[i,0]= frc[i,0]+ff*xr
				frc[i,1]= frc[i,1]+ff*yr
				frc[i,2]= frc[i,2]+ff*zr
				frc[j,0]= frc[j,0]-ff*xr
				frc[j,1]= frc[j,1]-ff*yr
				frc[j,2]= frc[j,2]-ff*zr
				#update the potential energy (shared between the two)
				en_pot[i] = en_pot[i] + 2*(r6i*(r6i-1)-ecut) #difference between potential energy and cutoff
				en_pot[j] = en_pot[j] + 2*(r6i*(r6i-1)-ecut)
	return frc,en_pot


#a function which uses the velocity verlet algorithm to integrate Newton's 
#equations of motion
def verlet_vel(frc,c_pos,c_vel,en_pot,en_kin):
	#initialise sums of velocites to be zero
	sumv_sqr=0

	#calculates previous position for use in verlet, using simple time step
	#backwards.
	p_pos = c_pos - (c_vel*deltat)

	#vectorised implementation of position update	
	new_pos = 2*c_pos - p_pos + frc*(deltat**2)
	#vectorised implementation of velocity update
	new_vel = (new_pos-p_pos)/(2*deltat)

	#sum of velocities/velocities squared
	sumv_sqr=sum(sum(new_vel**2))
	#temperature according to equiparition
	temp= sumv_sqr/((3*Npart)-3)

	#total energy per particle
	sum_en_pot = sum(en_pot)
	etot = (sum_en_pot+0.5*sumv_sqr)/Npart
	sum_en_kin = 0.5*sumv_sqr
	return new_pos, new_vel, temp,etot,sum_en_kin,sum_en_pot 

#It is possible to encounter an "explosion" with certain simulation
#times/speeds.To remedy this a position refolding function can be 
#introduced.
def refold(c_pos,init_dist,Npart,n_refolds):
	init_dist_sqr = init_dist**2
	for i in range(Npart-1):
		for j in range(i+1,Npart):
			xr = c_pos[i,0] - c_pos[j,0]
			yr = c_pos[i,1] - c_pos[j,1]
			zr = c_pos[i,2] - c_pos[j,2]
			r_sqr = xr**2 + yr**2 + zr**2
			#refold the positions (x-axis) depending on their displacement
			if r_sqr > init_dist_sqr and c_pos[i,0] > 0:
				c_pos[i,0] = c_pos[i,0] - 0.5*init_dist
				n_refolds = n_refolds+1
			if r_sqr > init_dist_sqr and c_pos[i,0] < 0:
				c_pos[i,0] = c_pos[i,0] + 0.5*init_dist
				n_refolds = n_refolds+1
			if r_sqr > init_dist_sqr and c_pos[j,0] > 0:
				c_pos[j,0] = c_pos[j,0] - 0.5*init_dist
				n_refolds=n_refolds+1
			if r_sqr > init_dist_sqr and c_pos[i,0] < 0:
				c_pos[j,0] = c_pos[j,0] + 0.5*init_dist
				n_refolds=n_refolds+1
	return n_refolds

#A velocity rescaling was also introduced, only required at certain
#simulation lengths/speeds.
def vel_rescale(ConstT,sum_en_kin,c_vel):
	#verify below equations, (Bussi, Donadio paper)
	scaling_factor = np.sqrt((3*ConstT/2)/sum_en_kin)
	k =1.38e-23
	rescaled=0
	if sum_en_kin > (3*k*ConstT)/2:
		c_vel = c_vel * scaling_factor
	return c_vel

#progress bar
def update_progress(workdone):
    print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone*100), end="", flush=True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SIMULATION BEGINS HERE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#print title
os.system('cls' if os.name == 'nt' else 'clear')
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~      MOLECULAR DYNAMICS SIMULATION      ~~~~~~")
print("~~~~~~     LENNARD-JONES CLUSTER COLLISION     ~~~~~~")
print("")
print("Created by: Callum Hutchinson.")
print("King's College London, 2016.")
print("")
print("Please enter requested values below.")
print("")
#Before we get started lets set some global simulation control variables.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~GLOBAL VARIABLES~~~~~~~~~#
print("Picosecond = 1e-12 s")
print("Femtosecond = 1e-15 s")
print("")
print('Enter time step:')
deltat = float(input()) #ps is 1e-12
print("")
print("Min. time steps =1000")
print("")
print('Enter number of time steps:')
Nsteps = int(input())
print("")
while Nsteps < 1000:
	print('Error: Min. timesteps = 1000')
	print('Enter number of timesteps:')
	Nsteps = int(input())

 #total of 5 picoseconds(5000)

ConstT = 20 #kelvin

#~~~~~~END OF GLOBAL VARIABLES~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#time initialised to zero
t=0

#number of refolds initialised at 0
n_refolds=0

#open the cluster co-ordinates from the external files
cluster1 ,cluster2 = get_clusters('cluster_size_70.dat','cluster_size_77.dat')

#request values for initial seperation and initial velocity
print(">5 for clusters fully apart")
print("")
print("Initial separation:")
init_sep = float(input())
print("")
print("(1e9 for merge within 5ns)")
print("")
print("Initial velocity:")
init_vel=float(input())
print("")
#initialise the cluster positions, velocities, forces...
c_pos,c_vel,frc,en_pot,en_kin,Npart,init_dist = init(cluster1,cluster2,init_sep,init_vel)
initial_pos = c_pos

#3d plot of initial positions
fig = plt.figure()
ax1 = fig.add_subplot(121,projection='3d')
ax1.set_title('Plot of initial cluster positions')
ax1.set_xlabel('x-axis')
ax1.set_ylabel('y-axis')
ax1.set_zlabel('z-axis')
ax1.scatter(c_pos[:70,0],c_pos[:70,1],c_pos[:70,2],c='b')
ax1.scatter(c_pos[70:,0],c_pos[70:,1],c_pos[70:,2],c='r')


#create a matrix/table to keep track of values,
# _______________________________________________
#| Temp  | E_tot   | E_kin  | E_pot |  Time      |
#|-------|---------|--------|-------|------------|
#|       |         |        |       |            |

values = np.zeros([int(Nsteps/1000)+1,5])
iter_number = 0

######################################
#####     CORE OF SIMULATION    ######
######################################

#actually evolve the system and call our functions

#start a timer to monitor our simulation times
start = time.time()
while t < deltat*Nsteps:
	update_progress(iter_number/Nsteps)

	new_frc,new_en_pot = get_force(c_pos,frc,en_pot,Npart)
	new_pos,new_vel,temp,etot,sum_en_kin,sum_en_pot = verlet_vel(new_frc,c_pos,c_vel,new_en_pot,en_kin)
	#n_refolds = refold(c_pos,init_dist,Npart,n_refolds)
	#c_vel = vel_rescale(ConstT,sum_en_kin,c_vel)

	#update values and introduce a fail-safe for if
	#there is an error and the cluster isnt moving
	if np.array_equal(c_pos,new_pos):
		print("Fatal Error: No movement!")
		quit()
	else:
		c_pos = new_pos
		c_vel = new_vel
		frc=new_frc
		en_pot=new_en_pot

	#add values to matrix for every 1000 iterations
	#program is terribly slow if you print these
	#values for every single timestep	
	if iter_number%1000 == 0:
		it_no = int(iter_number/1000)
		values[it_no,0] = temp
		values[it_no,1] = etot
		values[it_no,2] = sum_en_kin
		values[it_no,3] = sum_en_pot
		values[it_no,4] = t

	#increment iteration number and time
	iter_number = iter_number + 1
	t = t+deltat
	final_pos=c_pos

#end timer
end=time.time()
elapsed = end-start

#print some useful statistics
if np.array_equal(final_pos, initial_pos):
	print("Clusters haven't moved!")

#print(n_refolds)

#print time difference (down here for formating on screen)
os.system('cls' if os.name == 'nt' else 'clear')
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~       SIMULATION COMPLETE       ~~~~~~~~~~")
print("")
print("")
print("With ",deltat,"s time-steps, ",Nsteps," steps, an")
print("initial separation of ",init_sep," and an initial")
print("velocity of ",init_vel,", the simulation took: ",round(elapsed),"s.")
print("")
print("The final positions of the particles can be found in")
print("the file final_state.xyz")
print("")
print("The values, logged at every 1000 iterations can be")
print("found in the file final_values.txt in the format:")
print("")
print(" _______________________________________________")
print("| Temp  | E_tot   | E_kin  | E_pot |  Time      |")
print("|-------|---------|--------|-------|------------|")
print("|       |         |        |       |            |")
print("")
print("The matplotlib package should now be displaying a")
print("plot of the initial and final cluster positions.")
print("")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#3d plot of final positions
ax2 = fig.add_subplot(122,projection='3d')
ax2.set_title('Plot of final cluster positions')
ax2.set_xlabel('x-axis')
ax2.set_ylabel('y-axis')
ax2.set_zlabel('z-axis')
ax2.scatter(c_pos[:70,0],c_pos[:70,1],c_pos[:70,2],c='b')
ax2.scatter(c_pos[70:,0],c_pos[70:,1],c_pos[70:,2],c='r')
plt.show()



#saves values to txt file
np.savetxt('final_values.txt',values,delimiter=' ')

#write the final positions to an XYZ file (messy but works)
np.savetxt('final_state.xyz',c_pos,delimiter=' ')
data = pd.read_csv('final_state.xyz',header=None,delim_whitespace=True)
data.insert(0,'element','H')
data.to_csv('final_state.xyz',sep=' ',header=None,index=False)
def line_prepender(position,filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(int(position), 0)
        f.write(line.rstrip('\r\n') + '\n' + content)
        f.close()
line_prepender(1,'final_state.xyz','Values for the final positions of the clusters')
line_prepender(0,'final_state.xyz',str(Npart))















