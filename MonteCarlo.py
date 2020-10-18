import numpy as np

# Giving the lattice size and initial parameters
Lattice = (20,20)
J = 1.0
Lx, Ly = Lattice
num_sites = Lx * Ly
T = 1.0

# Numbering scheme for a 10 X 10 lattice
'''
  96              90
86              80
              70
67  68  69  60
  58  59  50  51  52  53  54  55  56  57  58  59
48  49  40  41  42  43  44  45  46  47  48  49
  39  30  31  32  33  34  35  36  37  38  39
29  20  21  22  23  24  25  26  27  28  29
  10  11  12  13  14  15  16  17  18  19
0   1   2   3   4   5   6   7   8   9
'''

# Nearest neighbour labels
right, left, top_right, top_left, bottom_right, bottom_left = (0,1,2,3,4,5)

N1 = Lx*(Ly-1)
num_nn = 6
nn_table = np.zeros((num_sites, num_nn),dtype=int)
for site in range(num_sites):
    nn_table[site,right] = site+1
    nn_table[site,left] = site-1
    nn_table[site,top_right] = site+Lx
    nn_table[site,top_left] = site+Lx-1
    nn_table[site,bottom_right] = site-Lx+1
    nn_table[site,bottom_left] = site-Lx
    
# Boundary conditions wrap
for site in range(0,num_sites,Lx):
    nn_table[site,left]=site+(Lx-1)
    nn_table[site,top_left]=site+Lx+Lx-1

for site in range(Lx-1,num_sites,Lx): nn_table[site,right]=site-(Lx-1)

for site in range(N1,num_sites):
    if site-N1+(Ly/2) < Lx: nn_table[site,top_right]=site-N1+(Ly/2)
    else: nn_table[site,top_right]=site-N1+(Ly/2)-Lx
    
for site in range(N1,num_sites):
    if site-N1-1+(Ly/2) < Lx: nn_table[site,top_left]=site-N1-1+(Ly/2)
    else: nn_table[site,top_left]=site-N1+(Ly/2)-Lx-1

for site in range(Lx-1,num_sites,Lx):
    nn_table[site,bottom_right]=site-Lx+1-Lx

for site in range(Lx):
    if site+num_sites-(Ly/2) < num_sites: nn_table[site,bottom_left]= site+num_sites-(Ly/2)
    else: nn_table[site,bottom_left]= site+num_sites-(Ly/2)-Lx
    
for site in range(Lx):
    if site+num_sites-(Ly/2)+1 < num_sites: nn_table[site,bottom_right]=site+num_sites-(Ly/2)+1
    else: nn_table[site,bottom_right]=site+num_sites-(Ly/2)-Lx+1

# This scheme can be cross checked by printing
#for s in range(num_sites):
#	print(s, ' ', nn_table[s,bottom_left], ' ', nn_table[s, bottom_right])
    

# Basis state
basis_state = np.ones((num_sites), dtype=int)

def initialize_basis_state():
	# Generating random initial state
	for i in range(num_sites):
		if np.random.uniform(0, 1.0) < 0.5: basis_state[i] = +1
		else: basis_state[i] = -1
	return None

def gen_next_state(beta):
	select_site = np.random.randint(0,num_sites)
	isum = 0
	for site in nn_table[select_site]:
		isum += basis_state[site]
	delE = 2.0 * J * basis_state[select_site] * isum
    
	# Transition probability
	W = np.exp(-beta*delE)

	# Acceptance
	if (np.random.uniform(0,1.0) < W):
		basis_state[select_site] = -basis_state[select_site]
	return None

def do_mcstep(beta):
	for i in range(num_sites): gen_next_state(beta)

def get_energy():
    isum = 0
    for i in range(num_sites):
        j = nn_table[i,right]
        isum += basis_state[i]*basis_state[j]
        j = nn_table[i,bottom_right]
        isum += basis_state[i]*basis_state[j]
        j = nn_table[i,top_right]
        isum += basis_state[i]*basis_state[j]
    return float(-J*isum)

def get_magnetization():
    isum = 0
    for i in range(num_sites):
        isum += basis_state[i]
    return float(abs(isum))


def run_simulation(T):
    beta = 1.0/T
    
    measuring_samples = 5000
    warmup_steps = 2000
    interval = 3
    count = interval
    samples = 0
    iwork = 0
    magn_sum = 0.0
    magn_sq_sum = 0.0
    magn_4_order = 0.0
    energy_sum = 0.0
    energy_sq_sum = 0.0
    energy_4_order = 0.0

    initialize_basis_state()
    for t in range(warmup_steps): do_mcstep(beta)

    while samples != measuring_samples:
        do_mcstep(beta)
		# Calcuate |M| and E 
        if count == interval:
            
            magn = get_magnetization()
            magn_sum += magn
            magn_sq_sum += magn**2
            magn_4_order += magn**4
            
            energy = get_energy()
            energy_sum += energy
            energy_sq_sum += energy**2
            energy_4_order += energy**4
            samples += 1
            count = 0
        count += 1
        
	# Printing progress
        work = 100.0*float(samples)/measuring_samples
        if (int(work) > iwork and int(work)%10==0):
            iwork = int(work);
            print("done = ", int(work),'%')
    print('simulation done!')

    # Averages
    magn_mean = float(magn_sum/samples)
    magn_sq_mean = float(magn_sq_sum/samples)
    magn_4_mean = float( magn_4_order/samples)
    
    energy_mean = float(energy_sum/samples)
    energy_sq_mean = float(energy_sq_sum/samples)
    energy_4_mean = float(energy_4_order/samples)
    
    # Calculations
    C = (energy_sq_mean - (energy_mean)**2)/(num_sites*(T**2))
    U_TL = 1 - float(magn_4_mean/(3 * magn_sq_mean**2))
    V_TL = 1 - float(energy_4_mean/(3 * energy_sq_mean**2))

    # Error bars
    magn_err = np.sqrt(magn_sq_mean-magn_mean*magn_mean)/(samples-1)
    energy_err = np.sqrt(energy_sq_mean-energy_mean*energy_mean)/(samples-1)
    
    
    return dict(magn=magn_mean, magn_err=magn_err, energy=energy_mean, energy_err=energy_err, heat_capacity=C, magn_cumulant=U_TL, energy_cumulant=V_TL )

file = open('assignment_simulation_results.txt','w')
file.write('# Monte Carlo Simulation\n')
file.write('# Ising Model, Triangular Lattice, Size {:d}x{:d}\n'.format(Lx,Ly))
file.write('#T         Magn       Err      Energy    Err      Heat Capacity     MagnCumulant      EnergyCumulant\n')

# Running the steps for different temperature values     
temp = []
temp_inv = []
C_ar = []
U_TL_ar = []
V_TL_ar = []

for T in np.arange(0.2,6.2,0.2):
    temp.append(T)
    temp_inv.append(float(1.0/T))
    results = run_simulation(T)
    C_ar.append(results['heat_capacity'])
    U_TL_ar.append(results['magn_cumulant'])
    V_TL_ar.append(results['energy_cumulant'])
    print('{:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f}'.format(T, results['magn'], results['magn_err'], results['energy'], results['energy_err'],results['heat_capacity'],results['magn_cumulant'],results['energy_cumulant']))
    file.write('{:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f}\n'.format(T, results['magn'], results['magn_err'], results['energy'], results['energy_err'],results['heat_capacity'],results['magn_cumulant'],results['energy_cumulant']))
file.close()


# Generating plots
import matplotlib.pyplot as plt

plt.plot(temp,C_ar,marker='o',markersize=6)
plt.title('Heat Capacity vs Temperature')
plt.xlabel('Temperature')
plt.ylabel('Heat Capacity per spin')
plt.savefig('Heat_capacity.png')
plt.show()
    
plt.plot(temp_inv,U_TL_ar,marker='o',markersize=6)
plt.xlim(0.2,0.4) 
# This limit was manually chosen to focus on the cumulant crossing point
plt.title('Order parameter cumulant vs Inverse temperature')
plt.xlabel('Inverse Temperature')
plt.ylabel('Order parameter cumulant')
plt.savefig('Mag_cumulant_inv.png')
plt.show()

plt.plot(temp,U_TL_ar,marker='o',markersize=6)
plt.title('Order parameter cumulant vs Temperature')
plt.xlabel('Temperature')
plt.ylabel('Order parameter cumulant')
plt.savefig('Mag_cumulant.png')
plt.show()

plt.plot(temp,V_TL_ar,marker='o',markersize=6)
plt.xlim(3.0,4.0)
plt.title('Energy cumulant vs Temperature')
plt.xlabel('Temperature')
plt.ylabel('Energy cumulant')
plt.savefig('Energy_cumulant.png')
plt.show()
