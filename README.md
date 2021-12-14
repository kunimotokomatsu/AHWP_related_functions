
# AHWP related functions in lib_KK.py  
example.py makes a plot of polarization efficiency and phase for two type of AHWP calculated with ignoring any reflection effects.  
The formalism in these functions can be found in [K. Komatsu et al. (2021)](http://dx.doi.org/10.1117/1.JATIS.7.3.034005) [[arXiv](https://arxiv.org/abs/2105.05561)] and [Thomas Essinger-Hileman (2014)](https://doi.org/10.1364/AO.52.000212) [[arXiv](https://arxiv.org/abs/1301.6160)].

## rms(x)  
**Purpose:**  
Calculate RMS  


<details><summary>Input and Return</summary>

**Input:**     
	x: one-dimensional array  
	
**Return:**   
	RMS of x  
</details> 

## Get_trans(air,air_err,sample,sample_err)  
**Purpose:**  
Calculate transmittance and its error   


<details><summary>Input and Return</summary>

**Input:**     
	air: air data (intensity w/o sample)  
	air: measurement error of air data    
	air: sample data (intensity w/ sample)  
	air: measurement error of sample data   
	
**Return:**   
	transmittance, error of transmittance  
</details> 

## read_txt2f(filename)  
**Purpose:**  
	Read a .txt, .dat, and so on have two columns  
	
<details><summary>Input and Return</summary>

**Input:**     
	filename: file name in strings  
	
**Return:**   
	Two array correspond to each column: Column-1, Column-2  
</details> 

## read_txt3f(filename)
**Purpose:**  
	Read a .txt, .dat, and so on have three columns  
	
<details><summary>Input and Return</summary>

**Input:**     
	filename: file name in strings  
	
**Return:**   
	Two array correspond to each column: Column-1, Column-2, Column-3  
</details> 

## read_txt4f(filename)
**Purpose:**  
	Read a .txt, .dat, and so on have four columns  
	
<details><summary>Input and Return</summary>

**Input:**     
	filename: file name in strings  
	
**Return:**   
	Two array correspond to each column: Column-1, Column-2, Column-3, Column-4  
</details> 

## fit_func(theta,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16)
**Purpose:**  
	Obtain a fit function of a modulated signal up to 8th harmonics of a rotational frequency  
	
<details><summary>Input and Return</summary>

**Input:**     
	theta: rotation angles of a HWP as an array  
	The coefficients a0 to a16 are parameters for fitting  
	a0: amplitude of DC part  
	a1: amplitude of 1st harmonics of the HWP rotation  
	a2: phase [rad] of 1st harmonics of the HWP rotation  
	a3: amplitude of 2nd harmonics of the HWP rotation  
	a4: phase [rad] of 2nd harmonics of the HWP rotation  
	a5: amplitude of 3rd harmonics of the HWP rotation  
	a6: phase [rad] of 3rd harmonics of the HWP rotation  
	a7: amplitude of 4th harmonics of the HWP rotation  
	a8: phase [rad] of 4thharmonics of the HWP rotation  
	a9: amplitude of 5th harmonics of the HWP rotation  
	a10: phase [rad] of 5th harmonics of the HWP rotation  
	a11: amplitude of 6th harmonics of the HWP rotation  
	a12: phase [rad] of 6th harmonics of the HWP rotation  
	a13: amplitude of 7th harmonics of the HWP rotation  
	a14: phase [rad] of 7th harmonics of the HWP rotation  
	a15: amplitude of 8th harmonics of the HWP rotation  
	a16: phase [rad] of 8th harmonics of the HWP rotation  
	
**Return:**  
	A modulated signal as a function of rotation angle which includes harmonics up to 8th  

</details> 

## expi(x)
**Purpose:**  
	Calculate exponential x  
	
<details><summary>Input and Return</summary>

**Input:**     
		x: one-dimensional array or a constant  
	
**Return:**   
	Exponential x e^{-ix}   
</details> 

## ygrid()  
**Purpose:**  
	Get a Mueller matrix of a wire grid aligned to y-axis  
	
<details><summary>Input and Return</summary>

**Input:**     
		None  
	
**Return:**  
	A Mueller matrix of a wire grid aligned to y-axis  
</details> 

## xgrid()
**Purpose:**  
	Get a Mueller matrix of a wire grid aligned to x-axis  
	
<details><summary>Input and Return</summary>

**Input:**     
		None  
	
**Return:**  
	A Mueller matrix of a wire grid aligned to x-axis  
</details> 

## Jones_to_Mueller(J)
**Purpose:**  
	Convert a Jones matrix to a Mueller matrix  
	
<details><summary>Input and Return</summary>

**Input:**     
		J: Jones matrix   
	
**Return:**  
	A Mueller matrix converted from an input Jones matrix  

</details> 

## n_eff(no,ne,chi,n1,theta1)
**Purpose:**  
	Calculate an effective refractive index of a birefringent plate as Eq.17 in [Thomas Essinger-Hileman (2014)](https://doi.org/10.1364/AO.52.000212)  
	
<details><summary>Input and Return</summary>

**Input:**     
		no: refractive index for ordinary ray  
		ne: refractive index for extraordinary ray  
		chi: rotation angle [rad]   
		n1: refractive index of the material surrounding the birefringent plate (e.g. air)  
		theta1: incident angle [rad]  
	
**Return:**  
	An effective refractive index of a birefringent plate  
</details> 

## sin_refraction(n,n1,theta1)
**Purpose:**  
	Calculate a sine of a refraction angle  
	
<details><summary>Input and Return</summary>

**Input:**     
		n: refractive index of a plate  
		n1: refractive index of the material surrounding the plate (e.g. air)  
		theta1: incident angle [rad]  
	
**Return:**  
	A sine of a refraction angle  
</details> 

## cos_refraction(n,n1,theta1)
**Purpose:**  
	Calculate a cosine of a refraction angle  
	
<details><summary>Input and Return</summary>

**Input:**     
		n: refractive index of a plate  
		n1: refractive index of the material surrounding the plate (e.g. air)  
		theta1: incident angle [rad]  
	
**Return:**  
	A cosine of a refraction angle  
</details> 

## transfer_matrix_1plate(freq,no,ne,thickness,chi,n1,theta1)
**Purpose:**  
	Calculate a transfer matrix for a birefringent plate in [Thomas Essinger-Hileman (2014)](https://doi.org/10.1364/AO.52.000212)  
	This function can be used ONLY for parallel plates  
	This function includes multiple reflection effects  
	
<details><summary>Input and Return</summary>

**Input:**     
		freq: frequency [Hz] (one-dimensional numpy.array or a constant)  
		no: refractive index for ordinary ray  
		ne: refractive index for extraordinary ray  
		thickness: thickness of the birefringent plate   
		chi: rotation angle [rad]   
		n1: refractive index of the material surrounding the birefringent plate (e.g. air)  
		theta1: incident angle [rad]  
	
**Return:**  
	A transfer matrix for a birefringent plate   
</details> 

## transfer_matrix_multilayer(freq,no_arr,ne_arr,thickness_arr,chi_arr,n1,theta1)
**Purpose:**  
	Calculate a transfer matrix for stacked birefringent plates in [Thomas Essinger-Hileman (2014)](https://doi.org/10.1364/AO.52.000212)  
	This function can be used ONLY for parallel plates  
	This function includes multiple reflection effects  
	
<details><summary>Input and Return</summary>

**Input:**     
		freq: frequency [Hz] (one-dimensional numpy.array or a constant)  
		no_arr: refractive index of each plate for ordinary ray as an array  
		ne_arr: refractive index of each plate  for extraordinary ray as an array  
		thickness_arr: thickness [m] of each plate as an array  
		chi_arr: rotation angle [rad] of each plate as an array  
		n1: refractive index of the material surrounding the birefringent plate (e.g. air)  
		theta1: incident angle [rad]  
	
**Return:**  
	A transfer matrix for stacked birefringent plates  
</details> 

## Jones_matrix_multilayer(freq,no_arr,ne_arr,thickness_arr,chi_arr,n1,theta1)
**Purpose:**  
	Calculate a Jones matrix for stacked birefringent plates using a method in [Thomas Essinger-Hileman (2014)](https://doi.org/10.1364/AO.52.000212)  
	This function can be used ONLY for parallel plates  
	This function includes multiple reflection effects  
	This function can be used for fit the transmittance  
	
<details><summary>Input and Return</summary>

**Input:**     
		freq: frequency [Hz] (one-dimensional numpy.array or a constant)  
		no_arr: refractive index of each plate for ordinary ray as an array  
		ne_arr: refractive index of each plate  for extraordinary ray as an array  
		thickness_arr: thickness [m] of each plate as an array  
		chi_arr: rotation angle [rad] of each plate as an array  
		n1: refractive index of the material surrounding the birefringent plate (e.g. air)  
		theta1: incident angle [rad]  
	
**Return:**  
	A Jones matrix for stacked birefringent plates  
</details> 

## Mueller_matrix_multilayer(freq,no_arr,ne_arr,thickness_arr,chi_arr,n1,theta1)
**Purpose:**  
	Calculate a Mueller matrix for stacked birefringent plates from a Jones matrix in [Thomas Essinger-Hileman (2014)](https://doi.org/10.1364/AO.52.000212)  
	This function can be used ONLY for parallel plates  
	This function includes multiple reflection effects  
	
<details><summary>Input and Return</summary>

**Input:**     
		freq: frequency [Hz] (one-dimensional numpy.array or a constant)  
		no_arr: refractive index of each plate for ordinary ray as an array  
		ne_arr: refractive index of each plate  for extraordinary ray as an array  
		thickness_arr: thickness [m] of each plate as an array  
		chi_arr: rotation angle [rad] of each plate as an array  
		n1: refractive index of the material surrounding the birefringent plate (e.g. air)  
		theta1: incident angle [rad]  
	
**Return:**  
	A Mueller matrix for stacked birefringent plates   
</details> 

## IVA_multilayer(freq,hwp_angle,no_arr,ne_arr,thickness_arr,chi_arr,n1,theta1,a_in,p_in,pol)
**Purpose:**  
	Calculate an array of intensity vs. angle (IVA) for stacked birefringent plates  
	This function can be used ONLY for parallel plates  
	This function includes multiple reflection effects  	
	
<details><summary>Input and Return</summary>

**Input:**     
		freq: frequency [Hz] (one-dimensional numpy.array or a constant)  
		no_arr: refractive index of each plate for ordinary ray as an array  
		ne_arr: refractive index of each plate  for extraordinary ray as an array  
		thickness_arr: thickness [m] of each plate as an array  
		chi_arr: rotation angle [rad] of each plate as an array  
		n1: refractive index of the material surrounding the birefringent plate (e.g. air)  
		theta1: incident angle [rad]  
		a_in: angle between a incident polarization and a detector sensitive angle [rad]  
		p_in: ratio of polarization in a incidnet radiation   
		pol: polarization type p or s as a string  
	
**Return:**  
	A IVA for stacked birefringent plates   
</details> 

## insert_gaps(gap_arr,no_arr,ne_arr,thickness_arr,angle_arr,n_gap)
**Purpose:**  
	Insert a layer of arbitrary thickness and refractive index between each plate   
	
<details><summary>Input and Return</summary>

**Input:**     
		gap_arr: thickness [m] of each gap as an array has (number of layer -1) elements    
		no_arr: refractive index of each plate for ordinary ray as an array  
		ne_arr: refractive index of each plate  for extraordinary ray as an array  
		thickness_arr: thickness [m] of each plate as an array  
		angle_arr: rotation angle [rad] of each plate as an array  
	
**Return:**  
	Gap inserted arrays: no_arr,ne_arr,thickness_arr,angle_arr  
</details> 

## add_MFA_layers(MFAthickness,nofMFAlayer,no_arr,ne_arr,thickness_arr,chi_arr,n1)
**Purpose:**  
	Insert multilayer film approximation (MFA) layers on two boundaris of the plate   
	MFA layers has the linear refractive index change  
	
<details><summary>Input and Return</summary>

**Input:**     
		MFAthickness: total thickness [m] of all MFA layers  
		nofMFAlayer: number of MFA layers  
		no_arr: refractive index of each plate for ordinary ray as an array  
		ne_arr: refractive index of each plate  for extraordinary ray as an array  
		thickness_arr: thickness [m] of each plate as an array  
		chi_arr: rotation angle [rad] of each plate as an array  
		n1: refractive index of the material surrounding the birefringent plate (e.g. air)  
	
**Return:**  
	MFA layers inserted arrays: no_arr,ne_arr,thickness_arr,angle_arr  
</details> 

## poleff_phase_multilayer(freqin,no_arr,ne_arr,thickness_arr,chi_arr,n1,theta1,a_in,p_in,pol)
**Purpose:**  
	Calculate amplitudes and phases of DC part and 4th harmonics in a modulated signal   
	This function can be used ONLY for parallel plates  
	This function includes multiple reflection effects  
	
<details><summary>Input and Return</summary>

**Input:**     
		freq: frequency [Hz] (one-dimensional numpy.array or a constant)  
		no_arr: refractive index of each plate for ordinary ray as an array  
		ne_arr: refractive index of each plate  for extraordinary ray as an array  
		thickness_arr: thickness [m] of each plate as an array  
		chi_arr: rotation angle [rad] of each plate as an array  
		n1: refractive index of the material surrounding the birefringent plate (e.g. air)  
		theta1: incident angle [rad]  
		a_in: angle between a incident polarization and a detector sensitive angle [rad]  
		p_in: ratio of polarization in a incidnet radiation   
		pol: polarization type p or s as a string  
	
**Return:**  
	Amplitudes and phases of DC part and 4th harmonics in a modulated signal: amp_4f, phase_4f, amp_DC  
</details> 

## amp2_phase2_multilayer(freqin,no_arr,ne_arr,thickness_arr,chi_arr,n1,theta1,a_in,p_in,pol)
**Purpose:**  
	Calculate amplitudes and phases of DC part and 2nd harmonics in a modulated signal   
	This function can be used ONLY for parallel plates  
	This function includes multiple reflection effects  
	
<details><summary>Input and Return</summary>

**Input:**     
		freq: frequency [Hz] (one-dimensional numpy.array or a constant)  
		no_arr: refractive index of each plate for ordinary ray as an array  
		ne_arr: refractive index of each plate  for extraordinary ray as an array  
		thickness_arr: thickness [m] of each plate as an array  
		chi_arr: rotation angle [rad] of each plate as an array  
		n1: refractive index of the material surrounding the birefringent plate (e.g. air)  
		theta1: incident angle [rad]  
		a_in: angle between a incident polarization and a detector sensitive angle [rad]  
		p_in: ratio of polarization in a incidnet radiation   
		pol: polarization type p or s as a string  
	
**Return:**  
	Amplitudes and phases of DC part and 2nd harmonics in a modulated signal: amp_2f, phase_2f, amp_DC  
</details> 

## amp_phase_multilayer(freqin,no_arr,ne_arr,thickness_arr,chi_arr,n1,theta1,a_in,p_in,pol)
**Purpose:**  
	Calculate amplitudes and phases of DC part and harmonics up to 8th in a modulated signal   
	This function can be used ONLY for parallel plates  
	This function includes multiple reflection effects  
	
<details><summary>Input and Return</summary>

**Input:**     
		freq: frequency [Hz] (one-dimensional numpy.array or a constant)  
		no_arr: refractive index of each plate for ordinary ray as an array  
		ne_arr: refractive index of each plate  for extraordinary ray as an array  
		thickness_arr: thickness [m] of each plate as an array  
		chi_arr: rotation angle [rad] of each plate as an array  
		n1: refractive index of the material surrounding the birefringent plate (e.g. air)  
		theta1: incident angle [rad]  
		a_in: angle between a incident polarization and a detector sensitive angle [rad]  
		p_in: ratio of polarization in a incidnet radiation   
		pol: polarization type p or s as a string  
	
**Return:**  
	Array of amplitudes and phases and a covariance matrix  
	They are alinged as the coefficients of fit_func   
</details> 

## transfer_matrix_1plate_normal(freq,no,ne,thickness,chi,n1)
**Purpose:**  
	Calculate a transfer matrix for a birefringent plate in [Thomas Essinger-Hileman (2014)](https://doi.org/10.1364/AO.52.000212)  
	This function can be used ONLY for parallel plates and normal incident  
	This function includes multiple reflection effects  
	This function does not use numpy.dot to calculate faster  
	
<details><summary>Input and Return</summary>

**Input:**     
		freq: frequency [Hz] (one-dimensional numpy.array or a constant)  
		no: refractive index for ordinary ray  
		ne: refractive index for extraordinary ray  
		thickness: thickness of the birefringent plate   
		chi: rotation angle [rad]   
		n1: refractive index of the material surrounding the birefringent plate (e.g. air)  
	
**Return:**  
	A transfer matrix for a birefringent plate   
</details> 

## transfer_matrix_multilayer_normal(freq,no_arr,ne_arr,thickness_arr,chi_arr,n1)
**Purpose:**  
	Calculate a transfer matrix for stacked birefringent plates in [Thomas Essinger-Hileman (2014)](https://doi.org/10.1364/AO.52.000212)  
	This function can be used ONLY for parallel plates and normal incident  
	This function includes multiple reflection effects  
	This function does not use numpy.dot to calculate faster  
	
<details><summary>Input and Return</summary>

**Input:**     
		freq: frequency [Hz] (one-dimensional numpy.array or a constant)  
		no_arr: refractive index of each plate for ordinary ray as an array  
		ne_arr: refractive index of each plate  for extraordinary ray as an array  
		thickness_arr: thickness [m] of each plate as an array  
		chi_arr: rotation angle [rad] of each plate as an array  
		n1: refractive index of the material surrounding the birefringent plate (e.g. air)  
		theta1: incident angle [rad]  
	
**Return:**  
	A transfer matrix for stacked birefringent plates  

</details> 

## Jones_matrix_multilayer_normal(freq,no_arr,ne_arr,thickness_arr,chi_arr,n1)
**Purpose:**  
	Calculate a Jones matrix for stacked birefringent plates using a method in [Thomas Essinger-Hileman (2014)](https://doi.org/10.1364/AO.52.000212)  
	This function can be used ONLY for parallel plates and normal incident  
	This function includes multiple reflection effects  
	This function does not use numpy.dot to calculate faster  
	We assume a liniear polarization as a incident radiation  
	This function can be used for fit the transmittance  
	
<details><summary>Input and Return</summary>

**Input:**     
		freq: frequency [Hz] (one-dimensional numpy.array or a constant)  
		no_arr: refractive index of each plate for ordinary ray as an array  
		ne_arr: refractive index of each plate  for extraordinary ray as an array  
		thickness_arr: thickness [m] of each plate as an array  
		chi_arr: rotation angle [rad] of each plate as an array  
		n1: refractive index of the material surrounding the birefringent plate (e.g. air)  
	
**Return:**  
	A Jones matrix for stacked birefringent plates  
</details> 

## Jones_to_0f2f4f(freq,jones_arr,ain)
**Purpose:**  
	Convert Jones matrix to amplitudes and phases of DC part, 2nd harmonics, and 4th harmonics in a modulated signal  
	We assume a liniear polarization as a incident radiation  
	
<details><summary>Input and Return</summary>

**Input:**     
		freq: frequency [Hz] (one-dimensional numpy.array or a constant)  
		jones_arr: Jones matrix per frequency  
		ain: angle between a incident polarization and a detector sensitive angle [rad]  
	
**Return:**  
	Amplitudes and phases of DC part and 2nd harmonics in a modulated signal: amp_4f, phase_4f, amp_2f, phase_2f, amp_DC  
</details> 

## fastcal_2f4f_normal(freq,no_arr,ne_arr,thickness_arr,chi_arr,ain,n1)  
**Purpose:**  
	Calculate amplitudes and phases of DC part, 2nd harmonics, and 4th harmonics in a modulated signal   
	This function can be used ONLY for parallel plates and normal incident  
	This function includes multiple reflection effects  
	This function does not use numpy.dot to calculate faster  
	We assume a liniear polarization as a incident radiation  
	
<details><summary>Input and Return</summary>

**Input:**     
		freq: frequency [Hz] (one-dimensional numpy.array or a constant)  
		no_arr: refractive index of each plate for ordinary ray as an array  
		ne_arr: refractive index of each plate  for extraordinary ray as an array  
		thickness_arr: thickness [m] of each plate as an array  
		chi_arr: rotation angle [rad] of each plate as an array  
		ain: angle between a incident polarization and a detector sensitive angle [rad]  
		n1: refractive index of the material surrounding the birefringent plate (e.g. air)  
	
**Return:**  
	Amplitudes and phases of DC part, 2nd harmonics, and 4th harmonics in a modulated signal: amp_4f, phase_4f, amp_2f, phase_2f, amp_DC  
</details> 

## Mueller_matrix_multilayer_normal(freq,no_arr,ne_arr,thickness_arr,chi_arr,ain,n1)
**Purpose:**  
	Calculate a Mueller matrix of stacked birefringent plate   
	This function can be used ONLY for parallel plates and normal incident  
	This function includes multiple reflection effects  
	This function does not use numpy.dot to calculate faster  
	We assume a liniear polarization as a incident radiation  
	
<details><summary>Input and Return</summary>

**Input:**     
		freq: frequency [Hz] (one-dimensional numpy.array or a constant)  
		no_arr: refractive index of each plate for ordinary ray as an array  
		ne_arr: refractive index of each plate  for extraordinary ray as an array  
		thickness_arr: thickness [m] of each plate as an array  
		chi_arr: rotation angle [rad] of each plate as an array  
		ain: angle between a incident polarization and a detector sensitive angle [rad]  
		n1: refractive index of the material surrounding the birefringent plate (e.g. air)  
	
**Return:**  
	A Mueller matrix of stacked birefringent plate   

</details> 

## phase_adjust_nf(p,n)
**Purpose:**  
	Adjust phase of n-th harmonics  
	
<details><summary>Input and Return</summary>

**Input:**     
	p: phase [rad] array  
	n: number of harmonics  
	
**Return:**  
	An adjusted phase array   
</details> 

## phase_adjust_nf_w_amp_minus(p,n)
**Purpose:**  
	Adjust phase of n-th harmonics  
	When some amplitudes have minus values  
	
<details><summary>Input and Return</summary>

**Input:**     
	p: phase [rad] array  
	n: number of harmonics  
	
**Return:**  
	An adjusted phase array   
</details> 

## litebird_lft_old_band(freq,data)
**Purpose:**  
	Devide data by old LFT frequency bands [KEK wiki](https://wiki.kek.jp/pages/viewpage.action?pageId=108104267)  
	
<details><summary>Input and Return</summary>

**Input:**     
	freq: frequency [Hz] in one-dimensional array  
	data: data array to be devided  
	
**Return:**   
 	Array: band center frequency, devided frequency, devided data, bandwidth  
</details> 

## litebird_lft_band(freq,data)
**Purpose:**  
	Devide data by current LFT frequency bands [KEK wiki](https://wiki.kek.jp/pages/viewpage.action?pageId=108104267)  
	
<details><summary>Input and Return</summary>

**Input:**     
	freq: frequency [Hz] in one-dimensional array  
	data: data array to be devided  
	
**Return:**  
 	Array: band center frequency, devided frequency, devided data, bandwidth  
</details> 

## litebird_band(freq,data)
**Purpose:**  
	Devide data by old LiteBIRD frequency bands [KEK wiki](https://wiki.kek.jp/pages/viewpage.action?pageId=108104267)  
	
<details><summary>Input and Return</summary>

**Input:**     
	freq: frequency [Hz] in one-dimensional array  
	data: data array to be devided  
	
**Return:**   
 	Array: band center frequency, devided frequency, devided data, bandwidth  

</details> 

## DLPF(time,data,err,freq_cut,rotation_time)
**Purpose:**  
	Apply desital low pass filter to a time ordered data  
	This function completely cuts the frequencies upper the cut-off frequency  
	
<details><summary>Input and Return</summary>

**Input:**     
	time: time [sec] in one-dimensional array  
	data: data to be cut in one-dimensional array  
	err: error of data (not used)  
	freq_cut: cut-off frequency  
	rotation_time: time for once rotation  
	
**Return:**  
	Array after cut: time, data, error of data  

</details> 

## unit()
**Purpose:**  
	Get a unit matrix in Muller matrix   
	
<details><summary>Input and Return</summary>

**Input:**     
	None  
	
**Return:**  
	A unit matrix in Muller matrix   
</details> 

## Mueller_rot(theta)
**Purpose:**  
	Get a rotation matrix in Muller matrix   
	
<details><summary>Input and Return</summary>

**Input:**     
	theta: rotation angle [rad]   
	
**Return:**  
	A rotation matrix in Muller matrix  
</details> 

## Mueller_shwp_wo_refl(freq,no,ne,thickness)
**Purpose:**  
	Get a Muller matrix for a birefringent plate  
	Any reflections are ignored in this function  
	This function can be used ONLY for parallel plates and normal incident  

<details><summary>Input and Return</summary>

**Input:**     
	freq: frequency [Hz] (one-dimensional numpy.array or a constant)  
	no: refractive index for ordinary ray  
	ne: refractive index for extraordinary ray  
	thickness: thickness of the birefringent plate  
	
**Return:**  
	A Muller matrix for a birefringent plate  
</details> 

## fit_4f(rho,a0,a4,phi4)
**Purpose:**  
	Obtain a fit function of a modulated signal only DC part and 4th harmonics of a rotational frequency  
	
<details><summary>Input and Return</summary>

**Input:**     
	theta: rotation angles of a HWP as an array  
	The coefficients a0 to phi4 are parameters for fitting  
	a0: amplitude of DC part   
	a4: amplitude of 4th harmonics of the HWP rotation  
	phi4: phase [rad] of 4thharmonics of the HWP rotation  
	
**Return:**  
	A modulated signal as a function of rotation angle which includes 4th harmonics  
</details> 

## poleff_phase_wo_refl(freqin,no_arr,ne_arr,thickness_arr,angle_arr,a_in,p_in)
**Purpose:**  
	Calculate amplitudes and phases of DC part and 4th harmonics in a modulated signal by stacked birefringent plates 
	This function can be used ONLY for parallel plates and normal incident   
	Any reflections are ignored in this function  
	We assume the stacked birefringent plates surrounded by air  
	We also assume a liniear polarization as a incident radiation  
	
<details><summary>Input and Return</summary>

**Input:**     
		freq: frequency [Hz] (one-dimensional numpy.array or a constant)  
		no_arr: refractive index of each plate for ordinary ray as an array  
		ne_arr: refractive index of each plate  for extraordinary ray as an array  
		thickness_arr: thickness [m] of each plate as an array  
		angle_arr: rotation angle [rad] of each plate as an array   
		a_in: angle between a incident polarization and a detector sensitive angle [rad]  
		p_in: ratio of polarization in a incidnet radiation   
	
**Return:**  
	Amplitudes and phases of DC part and 4th harmonics in a modulated signal: amp_4f, phase_4f, amp_DC  

</details> 

## shwp_wo_refl_w_angle(freq,no,ne,thickness,angle)
**Purpose:**  
	Get a Muller matrix for a birefringent plate with a rotation angle  
	Any reflections are ignored in this function  
	This function can be used ONLY for parallel plates and normal incident   
<details><summary>Input and Return</summary>

**Input:**     
	freq: frequency [Hz] (one-dimensional numpy.array or a constant)  
	no: refractive index for ordinary ray  
	ne: refractive index for extraordinary ray  
	thickness: thickness of the birefringent plate
	angle: rotation angle [rad]  
	
**Return:**  
	A Muller matrix for a birefringent plate  
</details> 

## fastcal_4f_normal_wo_refl(freq,no_arr,ne_arr,thickness_arr,angle_arr,a_in,p_in)
**Purpose:**  
	Calculate amplitudes and phases of DC part, and 4th harmonics in a modulated signal by stacked birefringent plates  
	This function can be used ONLY for parallel plates and normal incident  
	This function does not use numpy.dot to calculate faster  
	We assume the stacked birefringent plates surrounded by air  
	We also assume a liniear polarization as a incident radiation  
	Any reflections are ignored in this function   
	
<details><summary>Input and Return</summary>

**Input:**     
		freq: frequency [Hz] (one-dimensional numpy.array or a constant)  
		no_arr: refractive index of each plate for ordinary ray as an array  
		ne_arr: refractive index of each plate  for extraordinary ray as an array  
		thickness_arr: thickness [m] of each plate as an array  
		angle_arr: rotation angle [rad] of each plate as an array  
		a_in: angle between a incident polarization and a detector sensitive angle [rad]  
		p_in: ratio of polarization in a incidnet radiation  
	
**Return:**  
	Amplitudes and phases of DC part and 4th harmonics in a modulated signal: amp_4f, phase_4f, amp_DC   
</details> 

## fastcal_4f_normal_wo_refl_2(freq,no_arr,ne_arr,thickness_arr,angle_arr,a_in,p_in)
**Purpose:**  
	Calculate amplitudes and phases of 4th harmonics in a modulated signal by stacked birefringent plates  
	This function can be used ONLY for parallel plates and normal incident  
	This function does not use numpy.dot to calculate faster  
	We assume the stacked birefringent plates surrounded by air  
	We also assume a liniear polarization as a incident radiation  
	Any reflections are ignored in this function   
	This function directly calculates amplitudes and phases of 4th harmonics from elements of Mueller matrix  
	I DO NOT CHECK WHETHER THIS FUNCTION WORK CORRECT OR NOT  
	
<details><summary>Input and Return</summary>

**Input:**     
		freq: frequency [Hz] (one-dimensional numpy.array or a constant)  
		no_arr: refractive index of each plate for ordinary ray as an array  
		ne_arr: refractive index of each plate  for extraordinary ray as an array  
		thickness_arr: thickness [m] of each plate as an array  
		angle_arr: rotation angle [rad] of each plate as an array  
		a_in: angle between a incident polarization and a detector sensitive angle [rad]  
		p_in: ratio of polarization in a incidnet radiation  
	
**Return:**  
	Amplitudes and phases of 4th harmonics in a modulated signal: amp_4f, phase_4f, amp_DC(No value)  
</details> 

## band_ave_poleff_tophat(poleff,phase)
**Purpose:**  
	Calculate a band averaged polarization efficiency and phase with a top-hat weight  
	
<details><summary>Input and Return</summary>

**Input:**     
		poleff: polarization efficiency  
		phase: phase [rad] (this phase corresponds to a8 in fit_func())  
	
**Return:**  
	A band averaged polarization efficiency and phase: poleff, phase, maximum phase variation  
</details> 

## band_ave_poleff(poleff,phase, weight_func)
**Purpose:**  
	Calculate a band averaged polarization efficiency and phase with a weight function  
	
<details><summary>Input and Return</summary>

**Input:**     
		poleff: polarization efficiency  
		phase: phase [rad] (this phase corresponds to a8 in fit_func())  
		weight_func: weight function, its total should be one   
	
**Return:**  
	A band averaged polarization efficiency and phase: poleff, phase, maximum phase variation  
</details> 

## sum_cos4f(amp,phase, weight_func)
**Purpose:**  
	Integrate the 4th harmonics part with a weight function  
	This function is named cos but the result is not depend on sine or cosine   
	
<details><summary>Input and Return</summary>

**Input:**     
		amp: amplitude of 4th harmonics   
		phase: phase [rad] (this phase corresponds to a8 in fit_func())  
		weight_func: weight function, its total should be one   
	
**Return:**  
	A integrated amplitude and phase: amp, phase, maximum phase variation   
 </details> 

## sum_cos2f(amp,phase, weight_func)
**Purpose:**  
	Integrate the 2nd harmonics part with a weight function  
	This function is named cos but the result is not depend on sine or cosine  
	
<details><summary>Input and Return</summary>

**Input:**     
		amp: amplitude of 4th harmonics   
		phase: phase [rad] (this phase corresponds to a4 in fit_func())   
		weight_func: weight function, its total should be one  
	
**Return:**  
	A integrated amplitude and phase: amp, phase, maximum phase variation   
</details> 

## sum_trig(amp, phase)
**Purpose:**  
	Integrate sine (or cosine) waves  
	
<details><summary>Input and Return</summary>

**Input:**     
		amp: amplitude of the waves   
		phase: phase of the waves [rad]  
	
**Return:**  
	A integrated amplitude and phase: amp, phase  

</details> 

## fastcal_2f4f_normal_2(freq,no_arr,ne_arr,thickness_arr,chi_arr,ain,n1)
**Purpose:**  
	Calculate amplitudes and phases of 2nd and 4th harmonics in a modulated signal by stacked birefringent plates  
	This function can be used ONLY for parallel plates and normal incident    
	This function does not use numpy.dot to calculate faster   
	We assume a liniear polarization as a incident radiation  
	Any reflections are ignored in this function   
	This function directly calculates amplitudes and phases of 2nd and 4th harmonics from elements of Mueller matrix  
	I DO NOT CHECK WHETHER THIS FUNCTION WORK CORRECT OR NOT  
	
<details><summary>Input and Return</summary>

**Input:**     
		freq: frequency [Hz] (one-dimensional numpy.array or a constant)  
		no_arr: refractive index of each plate for ordinary ray as an array  
		ne_arr: refractive index of each plate  for extraordinary ray as an array  
		thickness_arr: thickness [m] of each plate as an array  
		chi_arr: rotation angle [rad] of each plate as an array  
		ain: angle between a incident polarization and a detector sensitive angle [rad]  
		n1: refractive index of the material surrounding the birefringent plate (e.g. air)   
	
**Return:**  
	Amplitudes and phases of DC part, 2nd harmonics, and 4th harmonics in a modulated signal: amp_4f, phase_4f, amp_2f, phase_2f, amp_DC(No value)   
</details> 

After line 800 is the matrix based calculation version of Mueller_matrix_multilayer but they are commented out