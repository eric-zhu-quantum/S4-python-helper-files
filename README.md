# S4-python-helper-files
Helper files for computing optical spectra from S4 (Stanford Stratified Structure Solver) in a multi-threaded fashion

SpectralScan - Python Module that takes an S4 simulation object and obtains the optical spectrum for a wavelength range in a multithreaded fashion.  

EricFitter - Fits optical spectrum data (wavelength, transmission/reflection) and fits it to a Fano/Lorentzian peak.  


More info about S4 can be found here:
https://web.stanford.edu/group/fan/S4/

I compiled my S4 library from source after following the instructions from:
https://github.com/phoebe-p/S4
