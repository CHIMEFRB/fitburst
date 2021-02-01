# fitburst - About

`fitburst` offers a collection of Python structures for fitting two-dimensional dynamic spectra of dispersed radio pulses in terms of basic physical parameters. `fitburst` is currently able to model spectra that contain a single burst or multiple "sub-bursts", as is typically observed in repeating FRBs (e.g., [Hessels et al., 2019](https://ui.adsabs.harvard.edu/abs/2019ApJ...876L..23H/abstract)). 

In its most general configuration, `fitburst` will determine a best-fit model of the spectrum $S_i = A_i F_i T_i$ for the $i$th sub-burst, where $A_i$ is the global amplitude of the sub-burst signal and the other two terms are each functions of several parameters:

1. the term $F_i \equiv F_i(\gamma_i, r_i)$ encodes the spectral energy distribution of the $i$th burst, assuming a "running" power-law distribution where $\gamma_i$ is the spectral index, $r_i$ is the "running" of the spectral index, and $f_0$ is a fixed reference frequency;

$$
F_i = (f/f_0)^{-\gamma_i + r_i\ln{(f/f_0)}}
$$

2. the term $T_i \equiv T_i({\rm DM}, t_{{\rm arr},i}, w_i, \tau_0)$ characterizes the temporal shape of the $i$th sub-burst. In the frequency-dependent scattering timescale ($\tau_0$) is non-zero, then `fitburst` will fit a pulse broadening function [McKinnon et al., 2014](https://ui.adsabs.harvard.edu/abs/2014PASP..126..476M/abstract) as the temporal shape, where $\tau = \tau_0(f/f_0)^\alpha$ and 

$$
T_i = \frac{1}{2\tau}\exp{\bigg(\frac{w_i^2}{2\tau^2} - \bigg[\frac{(t({\rm DM}) - t_{{\rm arr}, i})}{\tau}\bigg]}\bigg)\bigg\{1 + {\rm erf}\bigg[\frac{t({\rm DM}) - (t_{{\rm arr}, i} + w_i^2/\tau)}{w_i\sqrt{2}}\bigg]\bigg\}
$$

where $t({\rm DM})$ is the arrival of the pulse with a non-zero dispersion measure (${\rm DM}$) and $w_i$ is the "intrinsic" width of the un-scattered (Gaussian) profile. If instead $\tau_0$ is sufficiently small (i.e., if scattering is negligible), then we instead model the temporal shape of the burst as a Gaussian distribution of widths $w_i$

