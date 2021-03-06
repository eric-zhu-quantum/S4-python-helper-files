#Fitter module

import numpy as np
import scipy.optimize as opt


#uses numpy (as np) functions: 
# A is a numpy array
# if A = [1,2,2,3,4,5,5,] --> returns [1,2,3,4,5] 
#as well as the indices of the elements in A used
# does NOT modify A
#returns (A[index], index)
def RemoveConsecutiveClones(A):
    C = A[1:] - A[0:-1]
    index = np.argwhere(C != 0)    
    index = np.append(index, np.array([len(A)-1]))
    return (A[index], index)



# usage 
#    dydx =  Derivative(xdata,ydata)
# both xdata and ydata are np.arrays()
# both are row vectors 
def Derivative(xdata,ydata):
    dydx = np.zeros(xdata.shape)
    N = len(dydx)

    dydx[0] = (ydata[1]-ydata[0])/(xdata[1] - xdata[0])     # 2-point rule
    dydx[1] = (ydata[2]-ydata[0])/(xdata[1+1] - xdata[1-1]) # 3-point rule
    
    dydx[N-2]=(ydata[N-1]-ydata[N-3])/(xdata[N-1] - xdata[N-3]) # 3-point rule
    dydx[N-1]= (ydata[N-1]-ydata[N-2])/(xdata[N-1] - xdata[N-2]) # 2-point rule
    
    if (N>4): # 5 point rule
        dydx[2:(-2)] = (-ydata[4:] + 8.*ydata[3:(-1)]-8*ydata[1:(-3)] + ydata[0:(-4)])/ \
        (-xdata[4:] + 8.*xdata[3:(-1)]-8*xdata[1:(-3)] + xdata[0:(-4)])

    return dydx



# usage: 
#   xmin,xmax = FindProperRange(xdata, ydata) 
#spits back the index ranges where we see at least 2 inflection points... 
def FindProperRange(xdata, ydata, debug = 0):
    dydx = Derivative(xdata, ydata)
    d2ydx2 = Derivative(xdata, dydx)
    
    #find where dydx = 0... 
    
    N = len(xdata) # number of data points
    
    # find the indices for which we have a 
    DerivPos2Neg = 1 + np.argwhere(  (dydx[0:(-2)] >0.) &  (dydx[2:] < 0.) )
    DerivNeg2Pos = 1 + np.argwhere(  (dydx[0:(-2)] <0.) &  (dydx[2:] > 0.) )
        
    DerivPos2Neg = np.squeeze(DerivPos2Neg)
    DerivNeg2Pos = np.squeeze(DerivNeg2Pos)
         
    
    indexmin = []
    indexmax = []
    if debug == 1:
        print('Which indices are candidate  max optima: ',DerivPos2Neg)
        print('Which indices are candidate  min optima: ',DerivNeg2Pos)
    for index in DerivPos2Neg:
        #look to the left
        #                 where does it go from concave up to    concave down
        leftindex = 1+np.argwhere(  (d2ydx2[0:(index-2)]>0.) &  (d2ydx2[2:(index)]<0.) )    
        leftindex = np.squeeze(leftindex)
        #look to the right:
        #                 where does it go from concave down to   concave up
        riteindex = 1+index+1+np.argwhere(  (d2ydx2[(index+1):(-2)]<0.) &  (d2ydx2[(index+3):]>0.) )
        riteindex = np.squeeze(riteindex)
        if debug:
            print(index, leftindex, riteindex)
        try:
            leftwidth = (index-np.max(leftindex))
            ritewidth = (np.min(riteindex) - index)
            maxwidth =  max(leftwidth,ritewidth)
            indexmin.append(np.max(leftindex)-3*maxwidth)
            indexmax.append(np.min(riteindex)+3*maxwidth)
        except:
            # do nothing
            print('')
            
    for index in DerivNeg2Pos:
        #look to the left
        #                 where does it go from concave down to    concave up
        leftindex = 1+np.argwhere(  (d2ydx2[0:(index-2)]<0.) &  (d2ydx2[2:(index)]>0.) )    
        leftindex = np.squeeze(leftindex)
        #look to the right:
        #                 where does it go from concave down to   concave up
        riteindex = 1+index+1+np.argwhere(  (d2ydx2[(index+1):(-2)]>0.) &  (d2ydx2[(index+3):]<0.) )
        riteindex = np.squeeze(riteindex)
        if debug:
            print(index, leftindex, riteindex)
        try:
            leftwidth = (index-np.max(leftindex))
            ritewidth = (np.min(riteindex) - index)
            maxwidth =  max(leftwidth,ritewidth)
            indexmin.append(np.max(leftindex)-3*maxwidth)
            indexmax.append(np.min(riteindex)+3*maxwidth)
        except:
            # do nothing
            print('')         
    indmin = np.min(indexmin) if (np.min(indexmin)>0) else 0
    indmax = np.max(indexmax) if (np.max(indexmax)<N) else (N-1)
    return indmin , indmax





# normalize the ydata and xdata, so that everything is between 0 and +/-1
# normalizes them by the mean and max value
#   probably better to normalize by the mean and the standard deviation 
def Normalize(xdata, ydata):
    # best to normalize all the data, for faster convergence
    xdata_mean = np.mean(xdata)
    xdata_max = np.max(np.abs(xdata-xdata_mean))
    #
    ydata_mean = np.mean(ydata)
    ydata_max = np.max(np.abs(ydata-ydata_mean))
    #
    xdata_norm = (xdata.copy() - xdata_mean)/xdata_max
    ydata_norm = (ydata.copy() - ydata_mean)/ydata_max
    #
    return xdata_norm, ydata_norm, xdata_mean, xdata_max, ydata_mean, ydata_max


#Fano fit!
# x0guess - you must provide a guess for where the 'center' of the Fano resonance is
# the function also gives out verbose output if you set debug = True
# usage:  lambda0, Linewidth, q, xth, yth, Amplitude = FitFano(xdata, ydata, x0guess, debug = False)
def FitFano(xdata, ydata, x0guess, debug = False):
    #
    #Model definition and Cost function definition
    Fano = lambda x, A, x0,Gamma,DC, DC1,q:       A*( (q*(x-x0) /(0.5*Gamma)+ 1)**2) / (((x-x0)/(0.5*Gamma))**2 + 1) + DC  + DC1*(x-x0)

    ModelFunc = lambda x, params: Fano(x, params[0], params[1],params[2],params[3], params[4], params[5])
    #params = [A,   x0,     Gamma,      DC,     DC1,    q]
          
    FirstElem = lambda Array: Array[0]

    xdata_norm, ydata_norm, xdata_mean, xdata_max, ydata_mean, ydata_max = Normalize(xdata, ydata)
    x0guess_norm = (x0guess - xdata_mean)/xdata_max

    CostFunc = lambda params: sum((ydata_norm - ModelFunc(xdata_norm, params))** 2) 


    ParamsTry = np.array(range(0,6)) * np.nan
    #params = [A,   x0,     Gamma,      DC,     DC1,    q]
    #          0    1       2           3       4       5
    ParamsTry[1] = x0guess_norm         #x0
    ParamsTry[2] = 0.1                # linewidth (nm) (xaxis)
    ParamsTry[3] = np.min(ydata_norm)# DC (yaxis)
    ParamsTry[4] = 0                  # DC1 (yaxis)
    ParamsTry[5] = 0                  # q: unitless varies from 0-1, asymm coeff
    #peak amplitude  (y-axis):
    ParamsTry[0] = FirstElem(ydata_norm[np.min(np.abs(xdata_norm-x0guess_norm)) 
                        == np.abs(xdata_norm-x0guess_norm)]) - ParamsTry[3]        
    
    
    disp_on = 1 if (debug) else 0	

    ParamsOpt = opt.fmin(CostFunc, ParamsTry,disp = disp_on)
    initcost = CostFunc(ParamsTry)
    mincost = CostFunc(ParamsOpt)
    if debug:
        print("init cost = % 2.4e, min cost = % 2.4e" %(initcost, mincost))

    lambda0 = ParamsOpt[1] * xdata_max + xdata_mean # convert back to appropriate units
    Linewidth= ParamsOpt[2] * xdata_max
    Amplitude = ParamsOpt[0]*ydata_max
    q = ParamsOpt[5]
    #
    xth = np.arange(np.min(xdata_norm),np.max(xdata_norm),0.01)
    yth = ModelFunc(xth,ParamsOpt)*ydata_max + ydata_mean
    xth = xth* xdata_max + xdata_mean
    #
    #print(xth)
    #print(yth)
    return lambda0, Linewidth, q, xth, yth, Amplitude



#Lorentzian fit!
# x0guess - you must provide a guess for where the 'center' of the Fano resonance is
# the function also gives out verbose output if you set debug = True
# usage:  lambda0, Linewidth, q, xth, yth, Amplitude = FitFano(xdata, ydata, x0guess, debug = False)
def FitLorentzian( xdata, ydata, x0guess, debug = False):
    #
    #Model definition and Cost function definition
    ModelFunc = lambda x, params:   ((params[1])/(1+((x-params[2])/(0.5*params[3]))**2)) + \
                params[0] + params[4]*(x-params[2])

    

    FirstElem = lambda Array: Array[0]
    #
    #
    xdata_norm, ydata_norm, xdata_mean, xdata_max, ydata_mean, ydata_max = Normalize(xdata, ydata, x0guess)
    x0guess_norm = (x0guess - xdata_mean)/xdata_max

    CostFunc = lambda params: sum((ydata_norm - ModelFunc(xdata_norm, params))** 2) 

    ParamsTry = np.array(range(0,5)) * np.nan
    ParamsTry[0] = np.mean(ydata_norm)# DC background (y-axis)
    #peak amplitude  (y-axis):
    ParamsTry[1] = FirstElem(ydata_norm[np.min(np.abs(xdata_norm-x0guess_norm)) 
                        == np.abs(xdata_norm-x0guess_norm)]) - ParamsTry[0]
    ParamsTry[2] = x0guess_norm
    ParamsTry[3] = 0.1  # linewidth (nm)
    ParamsTry[4] = 0  # linear background

    fmin_disp = 1 if debug else 0		
    ParamsOpt = opt.fmin(CostFunc, ParamsTry, disp=fmin_disp)
    initcost = CostFunc(ParamsTry)
    mincost = CostFunc(ParamsOpt)
    if debug:
        print("init cost = % 2.4e, min cost = % 2.4e" %(initcost, mincost))

    lambda0 = ParamsOpt[2] * xdata_max + xdata_mean # convert back to appropriate units
    Linewidth= ParamsOpt[3] * xdata_max
    #
    xth = np.arange(np.min(xdata_norm),np.max(xdata_norm),0.01)
    yth = ModelFunc(xth,ParamsOpt)*ydata_max + ydata_mean
    xth = xth* xdata_max + xdata_mean
    #
    #print(xth)
    #print(yth)
    return lambda0, Linewidth, xth, yth
