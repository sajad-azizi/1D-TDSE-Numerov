
import numpy as np
import matplotlib.pyplot as plt
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
plt.rcParams["font.size"] = "24"


fig, ax = plt.subplots()

fig.tight_layout()

ax.xaxis.set_minor_locator(AutoMinorLocator())

d0 = np.loadtxt('spectrum_A05T2.dat')
d_pt_2 = np.loadtxt('spectrumANA_duu_A05T2.dat')


axrange = range(0,1000)

ax.plot(d0[axrange,0],d0[axrange,2],lw=2,color='blue',label=r'TDSE')
ax.plot(d_pt_2[axrange,0],d_pt_2[axrange,2]*4.6,'--',lw=2,color='red',label=r'SVEA')
ax.legend(loc='upper right',frameon=False)


ax.set_xlabel(r"Energy $[\mathsf{a.u.}]$", fontsize='27')
ax.set_ylabel(r"electron spectrum $p(E)$", fontsize='27')

ax.text(0.05,0.0045,r'(a)', fontsize='27')


#xc = 1.469972355
#ax1.axvline(x=xc, color='k', linestyle='--', lw = 0.4)
#ax2.axvline(x=xc, color='k', linestyle='--', lw = 0.4)
#ax3.axvline(x=xc, color='k', linestyle='--', lw = 0.4)


plt.savefig('svea_tsde_zero_photon_A05T2_gauss.pdf', format='pdf', bbox_inches='tight', facecolor='none')
#plt.savefig('ground_5fs_1e16_hydo.svg', format='svg', bbox_inches='tight', facecolor='none')
plt.show()
