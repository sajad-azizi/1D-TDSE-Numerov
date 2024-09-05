
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('classic')
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator,LogLocator)
import matplotlib.colors as colors
import matplotlib.cbook as cbook
import matplotlib.ticker as ticker

import mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist import HostAxes
from mpl_toolkits.axisartist import GridHelperCurveLinear


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
#plt.rcParams["font.size"] = "24"


fig, ax = plt.subplots()
plt.tick_params(direction='inout',which='major', length=11)


#FLP pulse
d = np.loadtxt('spectrum_A1T1.dat')
#plt.figure(figsize=(14,9))
plt.plot(d[0:,0], d[0:,1],'-',lw=2, color='blue')
ax.set_yscale('log')
#plt.plot(d[0::20,0], d[0::20,1],'-o',markersize=4,lw = 1, label=r'$|1+3|^2$')

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.LogLocator())

#plt.xlim(-0.01,0.2)
#plt.ylim(0,10)
plt.ylabel(r'electron spectrum $p(E)$', fontsize='27')
plt.xlabel(r'Energy $[\mathrm{a.u.}]$', fontsize='27')

#F0=0.1

#plt.axvline(x=0.173, color='k', linestyle='--', lw=1)
#plt.annotate(r"", xy=(0.0, 1e-11), xytext=(0.173, 1e-11), arrowprops=dict(arrowstyle="<->"))
#plt.text(0.07,2e-11,r"$\Delta$", fontsize=14)#\Delta = 0.173
"""
#F0=1
plt.axvline(x=0.0715, color='k', linestyle='--', lw=1)
plt.axvline(x=0.1478, color='k', linestyle='--', lw=1)
plt.annotate(r"", xy=(0.0715, 0.0001), xytext=(0.1478, 0.0001), arrowprops=dict(arrowstyle="<->"))
plt.text(0.11,0.00015,r"$\Delta$", fontsize=14)#\Delta = 0.0763
"""
"""
#F0=5
plt.axvline(x=0.045, color='k', linestyle='--', lw=1)
plt.axvline(x=0.08962, color='k', linestyle='--', lw=1)
plt.axvline(x=0.13424, color='k', linestyle='--', lw=1)

plt.annotate(r"", xy=(0.045, 0.04), xytext=(0.08962, 0.04), arrowprops=dict(arrowstyle="<->"))
plt.text(0.065,0.07,r"$\Delta$", fontsize=15)# = 0.04462
"""

#plt.legend()
plt.savefig('tdse_A1_T1.pdf', format='pdf', bbox_inches='tight', facecolor='none')
plt.show()




