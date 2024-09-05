import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator,LogLocator)

plt.rcParams['savefig.dpi'] = 1200
plt.rcParams["figure.dpi"] = 100
plt.rcParams['text.usetex'] = True 
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'cm'

data = np.loadtxt('spectrum_A05T2.dat')
data2 = np.loadtxt('spectrumANA_duu_A05T2.dat')


fig, ax = plt.subplots(ncols=1)

ev = 27.211369
axrange = range(0,1000)

ax.plot(data[axrange,0]*ev,data[axrange,2],lw=2,color='blue',label=r'TDSE')
ax.plot(data2[axrange,0]*ev,data2[axrange,2]*4.6,'--',lw=2,color='red',label=r'SVEA')


ax.text(2.0,0.0045,r'(a)', fontsize='27')
ax.set_ylim(-0.001,0.036)


#ax.set_yscale('log')
ax.set_xscale('linear')
plt.tick_params(direction='inout',which='major', length=11)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.set_ylabel(r"electron spectrum $p(E)$", fontsize='24')
ax.set_xlabel(r"Energy $[\mathrm{eV}]$", fontsize='24')
ax.yaxis.set_ticks([0,0.01,0.02,0.03])
#ax.xaxis.set_ticks([0,0.5,1,1.5])
plt.yticks(fontsize=24)
plt.xticks(fontsize=24)

secax = ax.secondary_xaxis('top')
secax.xaxis.set_ticklabels([])
secax.xaxis.set_minor_locator(AutoMinorLocator())
secax.tick_params(direction='inout',which='major', length=11)
#secax.xaxis.set_ticks([0,0.5,1,1.5])
secay= ax.secondary_yaxis('right')
secay.yaxis.set_ticklabels([])
secay.yaxis.set_minor_locator(AutoMinorLocator())
secay.yaxis.set_ticks([0,0.01,0.02,0.03])
secay.tick_params(direction='inout',which='major', length=11)

plt.legend(frameon=False,fontsize='24');#, bbox_to_anchor=(0.7, 0.3))

plt.savefig('svea_tsde_zero_photon_A05T2_gauss.pdf', format='pdf', bbox_inches='tight', facecolor='none')
#plt.savefig('ground_5fs_1e16_hydo.svg', format='svg', bbox_inches='tight', facecolor='none')
plt.show()
