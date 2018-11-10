import pystan
import scipy.optimize as opt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import glob
import array
mpl.use("Agg")


from hashlib import md5

##################################################
## Setup
##################################################
sns.set()

def specrat_fit(freqs,fcl,fcs,domega,gamma=1.0,n=2.5):
    return domega+np.multiply(1.0/gamma,np.log10(1+np.power(np.divide(freqs,fcs),n*gamma)))-np.multiply(1.0/gamma,np.log10(1+np.power(np.divide(freqs,fcl),n*gamma)))

def stressdrop(M0, vs, fc, C=1.9):
	return (7.*M0/16)*((2*np.pi*fc)/(C*vs))**3

# Function to cache and load model
def StanModel_cache(filename, model_name=None, **kwargs):
    """Reads a .stan file and returns a model. Automatically
    caches a compiled model and checks if recompilation is
    necessary."""
    with open(filename,'r') as myfile:
        model_code = myfile.read()

    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm

##################################################
## Ground truth
N = 500 # Number frequency points (Time points T = 2N)
J = 2   # Number of spectral ratios ('Specrats')
L = 1   # Number of azimuthal bins

# Ground truth parameters
nvals=[1.0,1.0,1.0,1.0,1.0,1.5,1.5,1.5,1.5,1.5,2.0,2.0, 2.0,2.0,2.0,2.5,2.5,2.5,2.5,2.5,3.0,3.0,3.0,3.0, 3.0]
fci = 10.
fc1 = 0.5
n1  = 2.0
Mr  = 2.5
sig = 0.5 
ng = 2.0 #guessed n-value (changeable) 

M0=1.0e18 #true value of moment in N m
vs=4000 #S wave velocity in m/s

# Fixed in stan model...
stresstrue=stressdrop(M0, vs, fc1)
stressguess=[]
gi = 1.0; g1 = 1.0
guess_corners=[]
fc_smalls=[]
fcis = []
difference = []
guess_corners_small=[]
for nval in nvals:
	ni=nval
	n1=nval
	difference.append(ng-nval)
	fci=np.random.uniform(low=8.0, high=20.0)
	fc_smalls.append(fci)
	fcis.append(fci)

	# Spectral ratio model
	log_segf = lambda(f): Mr \
                   + np.log10(1+(f/fci)**(ni*gi))/gi \
                   - np.log10(1+(f/fc1)**(n1*g1))/g1

	# Generate data under ground truth
	F  = np.logspace(-1.5,1.5,N)
	lYtrue=np.array([log_segf(f) for f in F])
	lY1 = np.array([log_segf(f) for f in F]) + np.random.normal(scale=sig,size=N)

	# Gather data
	segf_dat = {'N': N,
            'J': J,
	    'L': L,
            'f': F,
           # 'ly':np.array([lY1,lY2]),
	    'ly': lY1,
	   # 'li': [1,1],
            'mu_Mr': [2.5,2.5]}

	####
	# fit model with least squares
	####

	res=opt.least_squares(lambda x: specrat_fit(F, x[0], x[1], Mr, g1, n1)-lY1, [0.1, 1.5])

	popt=res['x']
	print('corner from least squares: ' + str(popt[0]))
	guess_corners.append(popt[0])
	guess_corners_small.append(popt[1])

	##################################################
	## Fit the model
	##################################################
	#	sm  = StanModel_cache(filename='van_houtte.stan')
#	fit = sm.sampling(data=segf_dat, iter=2000, chains=4)

	##################################################
	## Report results
	##################################################
#	print(fit)
#fit.plot()
#	fit.plot(pars=['ni','fc1'])
#fit.plot(pars=['fc1','fci','n1','ni','Mr'])
#	plt.savefig('fit_'+str(nval)+'.png')
#plt.show()
#	fcis=np.mean(fit.extract(permuted=True)['fci'], axis=0)
#	fc1s=np.mean(fit.extract(permuted=True)['fc1'], axis=0)
#	nis=np.mean(fit.extract(permuted=True)['ni'], axis=0)
#	n1s=np.mean(fit.extract(permuted=True)['n1'],axis=0)
#	Mrs=np.mean(fit.extract(permuted=True)['Mr'],axis=0)
#	sigs=np.mean(fit.extract(permuted=True)['sig'],axis=0)
	fig,ax=plt.subplots(nrows=1,ncols=1)
	ax.semilogx(F, lY1, 'r', label='data')
	ax.semilogx(F, lYtrue, 'b', label='true spectral ratio')
	ax.semilogx(F,specrat_fit(F,popt[0], popt[1],Mr,gamma=1.0,n=2.0), 'g', label='fitted ratio')
	ax.legend()
	stressguess.append(stressdrop(M0, vs, popt[0]))
#	fig,ax=plt.subplots(nrows=len(segf_dat['ly']), ncols=1)
#	for i in range(0,len(segf_dat['ly'])):
#	    ax[i].semilogx(F, segf_dat['ly'][i])
#	    log_fit=lambda f: np.log(Mrs[i]) \
 #           +np.log(1+(f/fcis[i])**(nis[1]*gi))/gi \
  #          -np.log(1+(f/fc1s[i])**(n1s[1]*g1))/g1
#	    ax[i].semilogx(F,np.array([log_fit(f) for f in F]))
#plt.savefig('out_plot_test1'+str(i)+'.png')
	plt.savefig('out_plot_n'+str(nval)+'_fci'+str(fci)+'.png')
fig2,ax2=plt.subplots(nrows=1, ncols=1)
for i,nval in enumerate(nvals):
	ax2.scatter(nval, stressguess[i])
plt.title("N = ")
plt.xlabel("N values")
plt.ylabel("Stress Drop (guessed)")
ax2.plot(nvals, stresstrue*np.ones((len(nvals),)), label='true stress drop')
ax2.legend()
plt.savefig('n = .png')

fig4,ax4=plt.subplots(nrows = 1, ncols = 1)
for i,freq in enumerate(fcis):
	ax4.scatter(freq,stressguess[i])
plt.ylabel("Stress Drop (guessed)")
plt.xlabel("Fci")
ax4.plot(fcis,stresstrue * np.ones(len(stressguess)),label = 'true stress drop')
ax4.legend()
plt.savefig('fci.png')

fig3,ax3=plt.subplots(nrows = 1, ncols = 1)
for i, nval in enumerate(nvals):
	ax3.scatter(difference[i],stressguess[i])
plt.title("N (guessed) = ")
plt.xlabel("N (guessed) - N (true)")
plt.ylabel("Stress Drop (guessed)")
ax3.plot(difference,stresstrue*np.ones((len(nvals),)),label = 'true stress drop')
ax3.legend()
plt.savefig('difference.png')

