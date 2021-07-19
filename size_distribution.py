import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import scipy.stats as stats
import math
from scipy import integrate
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# pull data files
root = 'HELP_MSH/take2'
folders = ['30nm/replicate1','100nm/replicate1','500nm/replicate1','nobeads/replicate1']
file_name = 'exported2.csv'

# define the physical parameters for the system
n = 1.333
eta = 8.9e-4
theta = 173.*np.pi/180.
lam = 633. # nm
q = 4*np.pi*n*np.sin(theta/2.)/(lam) # nm^-1
kB = 1.38e-23 # J/K
T = 273.15+37. # K
q2 = np.float_power(q*1.e9,2.) # m^-2
colors = ['g','lightskyblue','gold','m']

# algorithm for solving for multimodal size distribution using correlation function
def multimodal(t,g,gamma,alpha_hi=2,alpha_lo=-4):
	Amat = np.exp(-2.*np.outer(t,gamma)) # try both -1. and -2. multiplied by exp(-gamma*t)
	Gmat = np.linalg.solve(Amat,g) 
	Pmat = np.diag(np.ones(len(t)))
	Pmat = Pmat + np.diag(np.ones(len(t)-2),k=2) + np.diag(np.full((1,len(t)-1),-2.)[0],k=1)
	x2 = np.zeros(100)
	y2 = np.zeros_like(x2)
	# create array of regularizer values alpha
	alpha = np.logspace(alpha_lo,alpha_hi,num=len(x2))
	x2s = np.zeros(len(alpha)-1)
	all_alpha = np.zeros((100,len(t)))
	cmap = plt.cm.get_cmap('viridis',len(x2))
	colors = [cmap(i) for i in range(len(x2))]
	g_transf = np.hstack((g,np.zeros(len(t))))
	for k,a in enumerate(alpha):
		P_transf = np.sqrt(a)*Pmat
		A_transf = np.vstack((Amat,P_transf))
		reg_nnls = LinearRegression(positive=True)
		y_pred_nnls = reg_nnls.fit(A_transf,g_transf).predict(A_transf)
		G_coef = reg_nnls.coef_
		x2[k] = np.dot(G_coef.T,G_coef)
		resid = np.dot(Amat,G_coef) - g
		y2[k] = np.dot(resid.T,resid)
		all_alpha[k,:] = G_coef
	y2new = y2 - np.min(y2)
	y2new = y2new/np.max(y2new)
	x2new = x2 - np.min(x2)
	x2new = x2new/np.max(x2new)
	for k in range(len(alpha)-1):
		x2s[k] = (x2new[k+1]-x2new[k])/(y2new[k+1]-y2new[k])
	x2s = x2s + 1.
	best_a = np.argmin(np.abs(x2s))
	return all_alpha,x2,y2,x2s,best_a,alpha

for w,folder in enumerate(folders):
	# define the file path where file is
	file_path = root + '/' + folder + '/' + file_name

	# define column names and extract data from csv file in pandas dataframe
	column_names = ['Record', 'Sample Name', 'Measurement Position',
		                    'G', 'time',
		                    'Distribution Fit Data', 'Distribution Fit Delay Times',
		                    'Cumulants Fit Data', 'Cumulants Fit Delay Times',
		                    'Derived Count Rate', 'Measured Intercept',
		                    'Measured Baseline']
	df = pd.read_csv(file_path, header=None, names=column_names)

	# initialize arrays for all rows
	g = []
	t = []
	tfit = []
	g1fit = []

	# define range to iterate through and iterate to collect scattering data
	intensities_records = range(2)
	for record in intensities_records:
		g.append(np.array(
		    [float(i) for i in df.iloc[record]['G'].split(',')]))
		t.append(np.array(
		    [float(i) for i in df.iloc[record]['time'].split(',')]))
		tfit.append(np.array(
		    [float(i) for i in df.iloc[record]['Distribution Fit Delay Times'].split(',')]))
		g1fit.append(np.array(
		    [float(i) for i in df.iloc[record]['Distribution Fit Data'].split(',')]))
	g = np.array(g) # shape = (12, 192)
	t = np.array(t)
	B = df.iloc[intensities_records]['Measured Baseline']
	Ie = df.iloc[intensities_records]['Derived Count Rate']
	point_pos = df.iloc[record]['Measurement Position']
	epos = df.iloc[intensities_records]['Measurement Position']

	# try multimodal distribution fit on data
	for n in range(1):
		g1vec = g[n,:]
		tvec = t[n,:]
		g1vec = g1vec/np.max(g1vec)
		diams = np.logspace(-4.,9.,num=len(tvec)) # in microns
		gamma = np.logspace(-12.,1.,num=len(tvec))
		[all_alpha,x2,y2,x2s,best_a,alpha] = multimodal(tvec,g1vec,gamma)
		
		best_distr = all_alpha[best_a,:]
		plt.plot(gamma,best_distr,ls='-',marker='.',color=colors[w],label=folder[0:-11])
		plt.xscale('log')
		plt.ylabel('G(gamma) (weight of each scatterer size)')
		plt.xlabel('Decay rate Gamma')

plt.legend()
plt.show()


# simulate a multimodal distribution to correlation function
sizes = np.array([0.1,1.0,10.])
gamma = (kB*T*q2*1.e6)/(3.*np.pi*eta*sizes*1.e6)
tau = t[0,:]
corr = np.sum(np.exp(-2.*np.outer(tau,gamma)),axis=1)
corr = corr/corr[0]

# simulate a distribution with Gaussian noise
sigma = np.array([0.01,0.01,0.05])
mu = np.array([0.1,1.0,10.0])
num = 30
d = np.zeros(num*len(mu))
weight = np.zeros_like(d)
for l,m in enumerate(mu):
	distr = np.random.normal(m,sigma[l],num)
	wgt = 1/(sigma[l]*np.sqrt(2*np.pi))*np.exp(-(distr-m)**2 / (2*sigma[l]**2))
	d[l*num:(l+1)*num] = distr
	weight[l*num:(l+1)*num] = wgt
gamma = (kB*T*q2*1.e6)/(3.*np.pi*eta*d*1.e6)
tau = t[0,:]
decay = np.exp(-2.*np.outer(tau,gamma))
corr = np.dot(decay,weight)
corr = corr/corr[0]

diams = np.logspace(-2.,2.,num=len(tau)) # in microns
gamma = (kB*T*q2*1.e6)/(3.*np.pi*eta*diams*1.e6)
[all_alpha,x2,y2,x2s,best_a,alpha] = multimodal(tau,corr,gamma)

best_distr = all_alpha[best_a,:]
# normalize the distributions so that the integral is 1
area_distr = 0.
for j in range(len(diams)-1):
	area_distr += (np.log(diams[j+1]) - np.log(diams[j]))*best_distr[j+1]
best_distr = best_distr/area_distr
plt.plot(diams,best_distr,'b.')
plt.plot(d,weight*np.max(best_distr)/np.max(weight),'r.')
plt.xscale('log')
plt.ylabel('G(gamma) (weight of each scatterer size)')
plt.xlabel('Scatterer Size (microns)')
plt.show()