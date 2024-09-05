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


data = np.loadtxt('spectrum_pi0.dat')
data2 = np.loadtxt('spectrum_pi2.dat')

ev = 27.211369

fig, ax = plt.subplots(ncols=1)

arange = range(0,8000)

ax.plot(data[arange,0]*ev,data[arange,2],lw=2, color='blue', label=r"$\phi = 0$")
ax.plot(data2[arange,0]*ev,data2[arange,2],lw=2, color='red', label=r"$\phi = \pi/2$")


#ax.set_yscale('log')
ax.set_xscale('linear')
plt.tick_params(direction='inout',which='major', length=11)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.set_ylabel(r"electron spectrum $p(E)$", fontsize='24')
ax.set_xlabel(r"Energy $[\mathrm{eV}]$", fontsize='24')
#ax.yaxis.set_ticks([1e-6,1e-4,1e-2,1e0])
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
#secay.yaxis.set_ticks([1e-6,1e-4,1e-2,1e0])
secay.tick_params(direction='inout',which='major', length=11)


#ax.text(1.4,0.8,r'(b)', fontsize='27')
plt.legend(frameon=False,fontsize='24', bbox_to_anchor=(0.7, 0.3))
plt.savefig('independ_zero_photon_phase.pdf', format='pdf', bbox_inches='tight', facecolor='none')
plt.show();
