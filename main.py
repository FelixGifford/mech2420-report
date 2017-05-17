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


s6061data = np.loadtxt("data/stress_strain_6061.txt", skiprows=1)
s7075data = np.loadtxt("data/stress_strain_7075.txt", skiprows=1)

plt.plot(s6061data[:,0],s6061data[:,1], linestyle="-", color='r')
plt.plot(s7075data[:,0],s7075data[:,1], linestyle="-", color='b')


#linear_elastic = [elastic_modulus * i + int for i in strain_vals]

#plt.plot(strain_vals[0:5500]+0.002, linear_elastic[0:5500])

plt.ylabel(r"$\sigma MPa$")
plt.xlabel(r"$\epsilon$")

plt.axis('auto')
#plt.annotate('$\epsilon_U$', xy=(0.02, 18.1), xytext=(0.02, 15),
#            arrowprops=dict(facecolor='black', shrink=0.05),
#            )

plt.show()