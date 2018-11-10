import pystan
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import glob
mpl.use("Agg")

from hashlib import md5

##################################################
## Setup
##################################################
sns.set()

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
fci = 10.
fc1 = 0.5
ni  = 2.0
n1  = 2.0
Mr  = 2.5
sig = 0.5

# Fixed in stan model...
gi = 2.0; g1 = 2.0

# Spectral ratio model
log_segf = lambda f: np.log(Mr) \
                   + np.log(1+(f/fci)**(ni*gi))/gi \
                   - np.log(1+(f/fc1)**(n1*g1))/g1


#Load data
#stanfile = 'van_houtte'
data = glob.glob('../data/CODAspecrats_*20070623T07*4.9.npz')
m0 = 1.12e17 #magnitude of master event
#data = [np.load('../data/CODAspecrats_20070623T072004_4.9_BIRAT2_EE_snr2.npz'), encoding = 'latin1')]
data = [np.load(dat) for dat in data]
data= [dat['specratios'] for dat in data]
data=[d for dat in data for d in dat]

# Generate data under ground truth
F  = np.logspace(-3,3,N)
#F = np.array([
#lY1  = np.array([log_segf(dat.specratio) for dat in data])# + np.random.normal(scale=sig,size=N)  
#lY2  = np.array([log_segf(dat) for dat in data])# + np.random.normal(scale=sig,size=N)
lY1 = np.array([log_segf(f) for f in F]) + np.random.normal(scale=sig,size=N)
#lY2 = np.array([log_segf(f) for f in F]) + np.random.normal(scale=sig,size=N)

# Gather data
segf_dat = {'N': N,
            'J': J,
	    'L': L,
            'f': F,
           # 'ly':np.array([lY1,lY2]),
	    'ly': lY1,
	   # 'li': [1,1],
            'mu_Mr': [2.5,2.5]}


segf = [d.segf for d in data] [0:50]
egfs=[d.egf for d in data]
egf_mags=[float(egf.split('_')[-1]) for egf in egfs]


data=[dat for i, dat in enumerate(data) if egf_mags[i] ==2.5]
egf_mag2=[egfmag for i, egfmag in enumerate(egf_mags) if egfmag == 2.5]

egf_mags=egf_mags
egf_mags=[0.439*egf_mag + 0.0689*egf_mag*egf_mag + 1.22 for egf_mag in egf_mags]

freqs = [dat.freqs for dat in data]
specrats = [dat.specratio for dat in data]
freqs = np.asarray(freqs)
specrats = np.asarray(specrats)

egf_mos=[1.5*egf_mag + 9.1 for egf_mag in egf_mags]
mu_Mr=[np.log(m0)-egf_mo for egf_mo in egf_mos]
mu_Mr=[Mr*np.ones_like(freqs[i]) for i, mu_Mr in enumerate(mu_Mr)]

# ##################################################
# ## Import the spectral ratio data
# ##################################################
# data      = np.load('../data/specrats_N.SATH.npz')
#freqs     = data['freq']
#specrats  = data['specratios'] #[dat.specrats for dat in data] 

#freqs = np.asarray(freqs)

#specrats = np.asarray(specrats)


#mu_Mr = np.asarray(mu_Mr).ravel()


#specrats = specrats/np.log10(np.e)

# ## DEBUG -- thin the data...
#gspecrats  = specrats[:10,:]

#freqs = range(0,50)
freqs = freqs[0:50]
segf_dat = {'N':len(freqs), 'J':len(specrats), 'f':freqs,'ly':freqs, 'mu_Mr': mu_Mr}

##################################################
## Fit the model
##################################################
sm  = StanModel_cache(filename='van_houtte.stan')
fit = sm.sampling(data=segf_dat, iter=2000, chains=4)

##################################################
## Report results
##################################################
print(fit)
#fit.plot()
fit.plot(pars=['ni','n1'])
#fit.plot(pars=['fc1','fci','n1','ni','Mr'])
plt.savefig('n1.png')
#plt.show()
fcis=np.mean(fit.extract(permuted=True)['fci'], axis=0)
fc1s=np.mean(fit.extract(permuted=True)['fc1'], axis=0)
nis=np.mean(fit.extract(permuted=True)['ni'], axis=0)
n1s=np.mean(fit.extract(permuted=True)['n1'],axis=0)
Mrs=np.mean(fit.extract(permuted=True)['Mr'],axis=0)
sigs=np.mean(fit.extract(permuted=True)['sig'],axis=0)

fig,ax=plt.subplots(nrows=len(segf_dat['ly']), ncols=1)
for i in range(0,len(segf_dat['ly'])):
    ax[i].semilogx(F, segf_dat['ly'][i])
    log_fit=lambda f: np.log(Mrs[i]) \
            +np.log(1+(f/fcis[i])**(nis[1]*gi))/gi \
            -np.log(1+(f/fc1s[i])**(n1s[1]*g1))/g1
    ax[i].semilogx(F,np.array([log_fit(f) for f in F]))
#plt.savefig('out_plot_test1'+str(i)+'.png')
plt.savefig('out_plot_test1'+'.png')

