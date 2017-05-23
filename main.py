#!/Users/felixgifford/anaconda/bin/python3
import sys, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#mpl.rc('text', usetex = True)
#mpl.rc('font', **{'family':"sans-serif"})
#params = {'text.latex.preamble': [r'\usepackage{siunitx}']}   
#plt.rcParams.update(params)  

width = 12.5
height = 5
area = width*height

data_path = input("Path to data: ")
cross_section = float(input("Cross sectional area of Sample: "))
poisson = float(input("Enter Poisson ratio: "))

data = np.loadtxt(data_path, skiprows=1)

#order data by x axis (col 0)
#convert data to stress and strain
#take elastic modulus measurement

plt.plot(data[:,0],data[:,1], linestyle="-", color='r')
print(data)


plt.ylabel(r"$\sigma MPa$")
plt.xlabel(r"$\epsilon$")

plt.axis('auto')
#plt.annotate('$\epsilon_U$', xy=(0.02, 18.1), xytext=(0.02, 15),
#            arrowprops=dict(facecolor='black', shrink=0.05),
#            )

plt.show()