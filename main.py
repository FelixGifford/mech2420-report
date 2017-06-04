#!/Users/felixgifford/anaconda/bin/python3
print("Importing Modules")
import sys, os
print("System libraries")
import numpy as np
print("Numpy")
import scipy.stats
print("Scipy Statistics")
import matplotlib as mpl
print("Matplotlib")
import matplotlib.pyplot as plt
print("Pyplot")
#mpl.rc('text', usetex = True)
#mpl.rc('font', **{'family':"sans-serif"})
#params = {'text.latex.preamble': [r'\usepackage{siunitx}']}   
#plt.rcParams.update(params)  

width = 12.5
height = 5
area = width*height

data_path = "data/stress_strain_6061.txt" #input("Path to data: ")
cross_section = area #float(input("Cross sectional area of Sample: "))
poisson = 0.34 #float(input("Enter Poisson ratio: "))

raw_data = np.loadtxt(data_path, skiprows=1)

#order data by x axis (col 0)
sorted_data = raw_data[raw_data[:,0].argsort()]

#convert data to stress and strain
# data: [L+delta_0, P_0]
#		[L+delta_0, P_0] * [1/cross_section]
#		[........., ...]   [(delta_n-delta_1)/delta_n] <----
#		[L+delta_n, P_n]

d = (sorted_data * [1,1/cross_section])
d[:,0] = ((d[:,0]-d[0,0])/d[:,0])

#take elastic modulus measurement
elastic_modulus,i,r,p,err = scipy.stats.linregress(d[0:int(len(d)*0.2)])

plt.plot(d[:,0],d[:,1], linestyle="-", color='r')
plt.plot(d[:,0],elastic_modulus*d[:,0])
#print(data)

'''
plt.ylabel(r"$\sigma MPa$")
plt.xlabel(r"$\epsilon$")

plt.axis('auto')
#plt.annotate('$\epsilon_U$', xy=(0.02, 18.1), xytext=(0.02, 15),
#            arrowprops=dict(facecolor='black', shrink=0.05),
#            )
'''
plt.show()