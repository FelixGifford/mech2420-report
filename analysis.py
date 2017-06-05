#!/Users/felixgifford/anaconda/bin/python3
print("Importing Modules")
import sys, os
print("System libraries")
import configparser
print("Config file Parser")
import uncertainties
print("Uncertainties")
from uncertainties import unumpy
print("Uncertainties unumpy")
import numpy as np
print("Numpy")
import scipy.stats, scipy.interpolate, scipy.optimize
print("Scipy")
import matplotlib as mpl

mpl.use('pgf')
params = {
	"pgf.preamble": [
	r'\usepackage{siunitx}'
	]

}
mpl.rcParams.update(params)
print("Matplotlib")
import matplotlib.pyplot as plt
print("Pyplot")



class Sample(object):
	def __init__(self, header):
		self.sample = configparser.ConfigParser()

		self.sample.read(header)

		self.name = self.sample.get('Sample', 'name')
		self.id = self.sample.get('Sample', 'id')

		self.dimensions = (self.sample.getfloat('Values', 'width'), self.sample.getfloat('Values', 'height'))
		self.area = self.dimensions[0]*self.dimensions[1]

		self.poisson = self.sample.getfloat('Values', 'poisson')

		self.d_0 = uncertainties.ufloat(self.sample.getfloat('Values', 'd_zero'), self.sample.getfloat('Values', 'd_zero_error'))
		print(self.d_0)

		self.ss_path = self.sample.get('Data', 'ss_data')
		self.diff_path = self.sample.get('Data', 'diff_data')

	def createElastoPlastic_Model(self):
		self.ss_data = np.loadtxt(self.ss_path, skiprows=1)

		self.ss_data = self.ss_data[self.ss_data[:,0].argsort()]
		#convert the force column into stress
		#force is given in kN so e3 converts that to N
		self.ss_data *=  [1,(1e3)/self.area]
		#convert the length column into strain
		self.ss_data[:,0] = ((self.ss_data[:,0]-self.ss_data[0,0])/self.ss_data[:,0])

		self.ElastoPlastic_Model = ElastoPlastic_Model(self.ss_data)

	def create_SS_plot(self):
		self.ElastoPlastic_Model.create_plots()

		plt.title("Stress Strain relationship for {}".format(self.name))

		plt.savefig('pgf/{}_StressStrain.pgf'.format(self.id))

		plt.close()

	def createDiffractionModel(self):
		self.diff_data = np.loadtxt(self.diff_path, skiprows=1)

		self.DiffractionModel = DiffractionModel(self.diff_data, self.d_0)

		self.DiffractionModel.dump_to_disk(self.id)

	def create_diff_plot(self):
		self.DiffractionModel.create_plots()

		plt.title("Measured Strain {}".format(self.name))

		plt.savefig('pgf/{}_strain.pgf'.format(self.id))

		plt.close()



class ElastoPlastic_Model(object):
	'''Contains all the data and functions to generate an ElastoPlastic model from stress and strain data'''
	def __init__(self, ss_data):
		'''Initialize'''
		self.stress_strain_data = ss_data

		self.data_n = len(self.stress_strain_data)
		self.percent_x = lambda x: int(self.data_n*(x/100))

		self.create_model()

	def create_model(self):
		'''Creates the model functions from data'''
		self.stress = scipy.interpolate.interp1d(self.stress_strain_data[:,0], self.stress_strain_data[:,1])

		self.elastic_modulus = scipy.stats.linregress(self.stress_strain_data[0:self.percent_x(25)])

		stress_range = (self.stress_strain_data[0,1], self.stress_strain_data[-1,1])
		strain_range = (0.002, (stress_range[1]-stress_range[0])/self.elastic_modulus[0] + 0.002)

		self.E_02 = scipy.interpolate.interp1d(strain_range, stress_range)		#Find difference between Linear model and stress strain data
		
		x1 = max(strain_range[0], self.stress_strain_data[0,0])
		x2 = min(strain_range[1], self.stress_strain_data[-1,0])

		diff = lambda x: self.stress(x)-self.E_02(x)
		
		self.proof_strain = scipy.optimize.bisect(diff, x1, x2, xtol=0.0001)
		self.proof_stress = self.stress(self.proof_strain)

		#self.ultimate_stress = self.stress(self.percent_x(100))

	def create_plots(self):

		x=np.linspace(self.stress_strain_data[0,0], self.stress_strain_data[-1,0], 1000)
		stress_range = (self.stress_strain_data[0,1], self.stress_strain_data[-1,1])
		x1=np.linspace(0.002, (stress_range[1]-stress_range[0])/self.elastic_modulus[0] + 0.002, 1000)
		plt.plot(x, self.stress(x))
		plt.plot(x1, self.E_02(x1))
		plt.plot(self.proof_strain, self.proof_stress, 'kx')
		plt.annotate(r'$\sigma_{0.02}$', xy=(self.proof_strain, self.proof_stress), textcoords='offset points', xytext=(5,5) )

		plt.xlabel(r'$\epsilon$')
		plt.ylabel(r'$\sigma (\si{\pascal})$')

	def __str__(self):
		return "E: {:>5.2f} GPa, Proof Stress: {:>5.2f} MPa".format(self.elastic_modulus[0]*1e-9, self.proof_stress*1e-6)

class DiffractionModel(object):
	def __init__(self, diff_data, d_0):
		'''Initialize'''
		self.diff_data = diff_data
		self.d_0 = d_0
		self.data_n = len(self.diff_data)

		self.transformData()

	def transformData(self):
		self.axial = uncertainties.unumpy.uarray(self.diff_data[:,1], self.diff_data[:,2])
		self.transverse = uncertainties.unumpy.uarray(self.diff_data[:,3], self.diff_data[:,4])

		self.strain_measured = np.column_stack( (self.diff_data[:,0], (self.axial-self.d_0)/self.d_0, (self.transverse-self.d_0)/self.d_0 ) )

	def create_plots(self):
		print(self.strain_measured)
		fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True)
		ax0.errorbar(uncertainties.unumpy.nominal_values(self.strain_measured[1]),
					 uncertainties.unumpy.nominal_values(self.strain_measured[0]),
					 xerr=uncertainties.unumpy.std_devs(self.strain_measured[1]))
		ax1.errorbar(uncertainties.unumpy.nominal_values(self.strain_measured[2]),
					 uncertainties.unumpy.nominal_values(self.strain_measured[0]),
					 xerr=uncertainties.unumpy.std_devs(self.strain_measured[2]))

		plt.show()

	def dump_to_disk(self, id):
		np.savetxt("data/{}_strain.txt".format(id), self.strain_measured, delimiter=' & ', fmt='%5.15r', newline=' \\\\\n')
	#def __str__(self):
	#	return self.strain_measured

for sample in ["data/6061.cfg", "data/7075.cfg"]:
	s = Sample(sample)
	s.createElastoPlastic_Model()
	s.createDiffractionModel()
	print(s.ElastoPlastic_Model)
	print(s.DiffractionModel)
	s.create_diff_plot()

'''d_0 = uncertainties.ufloat(1.2181161, 8e-6)

diffraction_data = np.loadtxt("data/diffraction_data_6061.txt", skiprows=1)

#plt.plot(d[:,0],d[:,1], linestyle="-", color='r')
#plt.plot(d[:,0],elastic_modulus*d[:,0])
#print(data)
'''
'''
plt.ylabel(r"$\sigma MPa$")
plt.xlabel(r"$\epsilon$")

plt.axis('auto')
#plt.annotate('$\epsilon_U$', xy=(0.02, 18.1), xytext=(0.02, 15),
#            arrowprops=dict(facecolor='black', shrink=0.05),
#            )
'''
#plt.show()