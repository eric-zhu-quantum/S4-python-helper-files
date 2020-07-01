from multiprocessing import Pool, Process
import multiprocessing
import numpy as np

#divide up the the wavelength window into N segments
# where N = NumThreads
def DivideWavelengths(startwl, stopwl, res, NumThreads):
    wls = np.arange(startwl,stopwl, res)
    NumElem_wls = len(wls)
    startwl_ctr = 0

    wlsettings_arr = []
    for k in range(NumThreads):
        stopwl_ctr = min(int(np.floor((k+1)*(NumElem_wls+0.0)/float(NumThreads))),NumElem_wls-1)
        wlsettings = (wls[startwl_ctr], wls[stopwl_ctr], res)    
        #print(wlsettings)
        wlsettings_arr.append(wlsettings)
        startwl_ctr = stopwl_ctr+1
    #
    return wlsettings_arr 


#get the backreflection at a particular wavelength
# look only at reflection for now
# if we want transmission set the index to [0]
def ComputeSingleDataPoint(S, wl, whichlayer):
    S.SetFrequency(1/wl)
    return np.real(S.GetPowerFlux(whichlayer)[1])


#output is in parameter return_dict (we are passing by reference)
# for thread # procnum, compute the Transmission/Reflection for that 
# particular wavelength range
def ComputeSpectra(ArgsTuple,procnum, return_dict):
    S, wlsettings, whichlayer = ArgsTuple
    
    startwl, stopwl, res = wlsettings

    #ForwardData = []
    wls = np.arange(startwl, stopwl+res, res)
    Numwls = len(wls)
    BackwardData = wls*0j
    
    # for each wl in the wls array, compute Poynting Flux
    BackwardData=[ ComputeSingleDataPoint(S, wl,whichlayer) for wl in wls]    
    
    return_dict[procnum]= (procnum,BackwardData)


#this guy takes as input the simulation S, wavelength range, and number of threads
# and parcels them out to be calculated 
def MultiThreadedSpectralScan(S, startwl, stopwl, res, whichlayer, NumThreads):
    # create data to be inputted into the functions per thread... 

    Scopy = [S.Clone() for k in range(NumThreads)]


    wlsettings_arr = DivideWavelengths(startwl, stopwl, res, NumThreads)
    ArgsTupleArray = []

    for k in range(NumThreads):
        ArgsTupleArray.append((Scopy[k], wlsettings_arr[k], whichlayer))

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for k in range(NumThreads):
        p = multiprocessing.Process(target=ComputeSpectra, args=(ArgsTupleArray[k],k,return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    Parts =  return_dict.values()   

    #re-order the parts;
    for k in range(len(Parts)):
        for l in range(k):
            if (Parts[k][0] < Parts[l][0]): # swap
                NewPart = Parts[k]
                Parts[k] = Parts[l]
                Parts[l] = NewPart

    Spectrum = []
    for k in range(len(Parts)):
        index, data = Parts[k]
        #print(index)
        Spectrum = np.hstack([Spectrum,data])

    wls = []
    for wlsetting in wlsettings_arr:
        tmpwl = np.arange(wlsetting[0],wlsetting[1]+wlsetting[2], wlsetting[2])
        wls = np.hstack([wls,tmpwl])
        
    return wls, Spectrum

