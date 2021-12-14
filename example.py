import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
cmap = cm.get_cmap("jet")
from lib_KK import band_ave_poleff_tophat #(poleff,phase)
from lib_KK import phase_adjust_nf #(freq, phase, number of harmonics)
from lib_KK import litebird_lft_band #(freq, data)
from lib_KK import fastcal_4f_normal_wo_refl #(freq,no_arr,ne_arr,thickness_arr,chi_arr,ain,n1)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c=299792458. #m/s
pi=np.pi
e=np.e
radeg = (180./pi)
ep0 = 8.8542e-12 #s4 A2/m3 kg
u0=4.*pi*1e-7 #H/m=V/(A/s)/m
im=complex(0.,1.)
arcmin=1./60.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Values taken from http://dx.doi.org/10.1117/1.JATIS.7.3.034005
freq=np.linspace(1,200,200)*1.e9
no_arr=np.ones(5)*3.047
ne_arr=np.ones(5)*3.361
thickness_arr=np.ones(5)*4.9e-3
angle_arr1=np.array([22.67,133.63,0,-133.63,-22.67])/radeg
angle_arr2=np.array([88.65,61.68,0,61.68,88.65])/radeg
a_in=45./radeg
p_in=1.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
poleff1, phase1, DC1=fastcal_4f_normal_wo_refl(freq,no_arr,ne_arr,thickness_arr,angle_arr1,a_in,p_in)
poleff2, phase2, DC2=fastcal_4f_normal_wo_refl(freq,no_arr,ne_arr,thickness_arr,angle_arr2,a_in,p_in)

poleff1=2.*poleff1
poleff2=2.*poleff2

freq_per_band=litebird_lft_band(freq,freq)
poleff1_per_band=litebird_lft_band(freq,poleff1)[2]
poleff2_per_band=litebird_lft_band(freq,poleff2)[2]
phase1_per_band=litebird_lft_band(freq,phase1)[2]
phase2_per_band=litebird_lft_band(freq,phase2)[2]


band_ave_poleff1=[]
band_ave_poleff2=[]
band_ave_phase1=[]
band_ave_phase2=[]
for i in range(len(freq_per_band[0])):
    ave1=band_ave_poleff_tophat(poleff1_per_band[i],phase1_per_band[i])
    ave2=band_ave_poleff_tophat(poleff2_per_band[i],phase2_per_band[i])
    band_ave_poleff1.append(ave1[0])
    band_ave_poleff2.append(ave2[0])
    band_ave_phase1.append(ave1[1])
    band_ave_phase2.append(ave2[1])
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
color_list=cmap(np.linspace(0.,1.,2))

plt.rcParams.update({'font.size': 10})

print(freq_per_band[-1])

fig = plt.figure(figsize=(10,5))
gs_master = gridspec.GridSpec(nrows=1,ncols=2,height_ratios=[1])
ax1=fig.add_subplot(gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0, 0])[:])
ax2=fig.add_subplot(gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0, 1])[:])

ax1.set_title('5 layers AHWP')
ax2.set_title('Design taken from http://dx.doi.org/10.1117/1.JATIS.7.3.034005',fontsize=10)

ax1.plot(freq*1.e-9,poleff1,linewidth=1,color=color_list[0],label='Anti-symmetric design')
ax1.plot(freq*1.e-9,poleff2,linewidth=1,color=color_list[1],label='Symmetric design')
ax1.scatter(freq_per_band[0]*1.e-9,band_ave_poleff1,marker='.',color=color_list[0],label='Anti-symmetric design (band average)')
ax1.scatter(freq_per_band[0]*1.e-9,band_ave_poleff2,marker='.',color=color_list[1],label='Symmetric design (band average)')
ax1.set_xlabel('Frequency [GHz]',fontsize=12)
ax1.set_ylabel('Polarization efficiency $2D_{4}$',fontsize=12)
#plt.legend()
ax1.set_xlim([0,200])
ax1.set_ylim([0.,1.05])
ax1.grid()

ax2.plot(freq*1.e-9,phase_adjust_nf(phase1,4)*radeg,linewidth=1,color=color_list[0])
ax2.plot(freq*1.e-9,phase_adjust_nf(phase2,4)*radeg,linewidth=1,color=color_list[1])
ax2.scatter(freq_per_band[0]*1.e-9,np.array(band_ave_phase1)*radeg,marker='.',color=color_list[0])
ax2.scatter(freq_per_band[0]*1.e-9,np.array(band_ave_phase2)*radeg,marker='.',color=color_list[1])
ax2.set_xlabel('Frequency [GHz]',fontsize=12)
ax2.set_ylabel('Phase $\phi_{4}$ [deg.]',fontsize=12)
ax2.set_xlim([0,200])
#plt.ylim([-1+22.5,1+22.5])
ax2.grid()
#plt.subplots_adjust(hspace=0.1,wspace=0.1)

ax1.legend(bbox_to_anchor=(0., 1.2, 2., 0.05), loc='upper left', ncol=2, borderaxespad=0, mode='expand')
plt.tight_layout()
plt.savefig('example.png')
plt.show()
plt.clf()