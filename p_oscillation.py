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


data = np.loadtxt('spectrum_A15T2.dat')


fig, ax = plt.subplots(ncols=1)

ax.plot(data[:,0]*27.211369,data[:,1],lw=2, color='k',zorder = 1)

ax.fill_between(data[:,0]*27.211369,data[:,1], color = 'gray', alpha=.2,zorder = 0)

ax.set_yscale('log')
ax.set_xscale('linear')
plt.tick_params(direction='inout',which='major', length=11)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.set_xlabel(r"Energy $[\mathrm{eV}]$", fontsize='27')
ax.set_ylabel(r"electron spectrum $p(E)$", fontsize='27')
ax.yaxis.set_ticks([1e-6,1e-4,1e-2,1e0])
ax.xaxis.set_ticks([0,10,20,30,40])
plt.yticks(fontsize=24)
plt.xticks(fontsize=24)

secax = ax.secondary_xaxis('top')
secax.xaxis.set_ticklabels([])
secax.xaxis.set_minor_locator(AutoMinorLocator())
secax.tick_params(direction='inout',which='major', length=11)
secax.xaxis.set_ticks([0,10,20,30,40])
secay= ax.secondary_yaxis('right')
secay.yaxis.set_ticklabels([])
secay.yaxis.set_ticks([1e-6,1e-4,1e-2,1e0])
secay.tick_params(direction='inout',which='major', length=11)


ax.text(1.4,0.8,r'(b)', fontsize='27')

plt.savefig('osci_A15_T2.pdf', format='pdf', bbox_inches='tight', facecolor='none')
plt.show();
