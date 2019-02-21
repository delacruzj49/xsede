from PyEMD import EEMD
import numpy as np
import pylab as plt
import pandas as pd
import sys,string
from pyeeg import *
from pylab import *
from numpy import *

# Define signal
#t = np.linspace(0, 1, 200)

#sin = lambda x,p: np.sin(2*np.pi*x*t+p)
#S = 3*sin(18,0.2)*(t-0.2)**2
#S += 5*sin(11,2.7)
#S += 3*sin(14,1.6)
#S += 1*np.sin(4*2*np.pi*(t-0.8)**2)
#S += t**2.1 -t

S = np.loadtxt('20181109_Control.csv', delimiter=',', dtype='float', skiprows=1, unpack=True)
#noAct, control, piano, reading = loadtxt('20181102_4act.csv', delimiter=',', dtype='float', skiprows=1, unpack=True)
#S = control
t = np.arange(len(S))

# Assign EEMD to `eemd` variable
eemd = EEMD()

# Say we want detect extrema using parabolic method
emd = eemd.EMD
emd.extrema_detection="parabol"

# Execute EEMD on S
#eIMFs = eemd.eemd(S, t)
eIMFs = eemd.eemd(S)
nIMFs = eIMFs.shape[0]

# Plot results
plt.figure(figsize=(12,9))
plt.subplot(nIMFs+1, 1, 1)
plt.plot(t, S, 'r')

nFiltered = int(0.8*nIMFs)
filteredS = [sum(x) for x in zip(*eIMFs[range(nFiltered)])]
np.savetxt("filteredS.csv", S, delimiter=",")

for n in range(nIMFs):
    plt.subplot(nIMFs+1, 1, n+2)
    plt.plot(t, eIMFs[n], 'g')
    plt.ylabel("eIMF %i" %(n+1))
    plt.locator_params(axis='y', nbins=5)

plt.xlabel("Time [s]")
plt.tight_layout()
plt.savefig('eemd_example', dpi=120)
plt.show()



# Feature Extraction continues from this point on
DIM = 10
TAU = 4
Fs = 128 # sampling rate
Band = [0.5, 4, 7, 12, 30, 100] # EEG bands boundary points
#nBands = 5 # number of bands
Kmax = 5
window_size = 1 # how many seconds in a window
window = window_size * Fs
step_distance = 0.5 # how many seconds in the sliding distance
step = int(step_distance*window)

# def add2Bands(delta, theta, alpha, beta, gamma, deltaRIR, thetaRIR, alphaRIR, betaRIR, gammaRIR, psi, rir):
#    delta.append(psi[0])
#    theta.append(psi[1])
#    alpha.append(psi[2])
#    beta.append(psi[3])
#    gamma.append(psi[4])
#    deltaRIR.append(rir[0])
#    thetaRIR.append(rir[1])
#    alphaRIR.append(rir[2])
#    betaRIR.append(rir[3])
#    gammaRIR.append(rir[4])

S = np.loadtxt('filteredS.csv', delimiter=',', dtype='float', skiprows=1, unpack=True)
S = np.subtract(S, 0.6)
#print(len(S))

delta = list()
theta = list()
alpha = list()
beta = list()
gamma = list()
deltaRIR = list()
thetaRIR = list()
alphaRIR = list()
betaRIR = list()
gammaRIR = list()
sPFD = list()
sHFD = list()
hjorth_mobility = list()
hjorth_complexity = list()
spec_entropy = list()
ssvd_entropy = list()
sfisher_info = list()
approx_entropy = list()
sDFA = list()
sHurst = list()

data_len = len(S)
start_point = 0
end_point = window
while end_point <= data_len:
    segment = S[start_point:end_point]
#    result = bin_power(segment, Band, Fs)
    psi, rir = bin_power(segment, Band, Fs)
#    psi = result[0]
#    rir = result[1]
    delta.append(psi[0])
    theta.append(psi[1])
    alpha.append(psi[2])
    beta.append(psi[3])
    gamma.append(psi[4])
    deltaRIR.append(rir[0])
    thetaRIR.append(rir[1])
    alphaRIR.append(rir[2])
    betaRIR.append(rir[3])
    gammaRIR.append(rir[4])

    sPFD.append(pfd(segment))
    sHFD.append(hfd(segment, Kmax))
    hjorthM, hjorthC = hjorth(segment)
    hjorth_mobility.append(hjorthM)
    hjorth_complexity.append(hjorthC)
    spec_entropy.append(spectral_entropy(segment, Band, Fs))
    M = embed_seq(segment, TAU, DIM)
    W = svd(M, compute_uv=0)
    W /= sum(W)
    ssvd_entropy.append(svd_entropy(segment, TAU, DIM, W))
    sfisher_info.append(fisher_info(segment, TAU, DIM, W))
    R = std(segment) * 0.3
    approx_entropy.append(ap_entropy(segment, DIM, R))
    sDFA.append(dfa(segment))
    sHurst.append(hurst(segment))

    start_point += step
    end_point = start_point + window
#print(result[0])
#print(result[1])
features = pd.DataFrame(np.column_stack( \
    [delta, theta, alpha, beta, gamma, \
    deltaRIR, thetaRIR, alphaRIR, betaRIR, gammaRIR, \
    sPFD, sHFD, hjorth_mobility, hjorth_complexity, \
    spec_entropy, ssvd_entropy, sfisher_info, \
    approx_entropy, sDFA, sHurst]), \
    columns=['delta', 'theta', 'alpha', 'beta', 'gamma', \
    'deltaRIR', 'thetaRIR', 'alphaRIR', 'betaRIR', 'gammaRIR', \
    'PFD', 'HFD', 'hjorth_mobility', 'hjorth_complexity', \
    'spec_entropy', 'svd_entropy', 'fisher_info', \
    'approx_entropy', 'DFA', 'Hurst'])
np.savetxt("features.csv", features, delimiter=",", \
    header="delta,theta,alpha,beta,gamma, \
    deltaRIR,thetaRIR,alphaRIR,betaRIR,gammaRIR, \
    PFD,HFD,hjorth_mobility,hjorth_complexity, \
    spec_entropy,svd_entropy,fisher_info, \
    approx_entropy,DFA,Hurst", comments="")
