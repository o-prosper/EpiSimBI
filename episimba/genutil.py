# # Simulate CTMC for a general compartmental epi model


import sympy as sm
import numpy as np
from sympy import init_printing

def CTMC(data,init_cond,trans_rates,trans_matrix,param_vals,param_names):
    
    from copy import copy

    # Set up length of simulation and reporting times.
    T      = data.T
    tmin   = data.tmin
    tmax   = data.tmax
    maxiter = data.maxiter
    
    # Set state variables equal to initial condition vector
    x  = [init_cond] # This will be used to save the state values at the reporting times
    xt = np.array(init_cond) # This will be used to update the states at each event time
    
    # get first observation reporting time beyond tmin.
    j  = 1
    Treport = T[j]
    
    # start time
    t  = tmin
    ii = 0
    while (t < tmax) and (ii < maxiter):
        ii += 1    
        
        # calculate transition rates by substituting current values of the states
        # and parameters
        # 1. Concatenate the state values (except for C) and the parameter values
        v = np.concatenate([xt[:-1],param_vals]) 
        # 2. Create a dictionary for each variable x1:xn and parameter p1:pn
        values = dict([(key,val) for key,val in zip(param_names,v)])
        # 3. Evaluate the transition rates at these state and parameter values
        rates  = np.array([float(expr.subs(values)) for expr in trans_rates])

        # Sum the rates, and if it is positive, compute the cdf of the transition probabilities,
        # Choose the time of the next event and which event occurs,
        # Update the state vector and record the states at the reporting times.
        # Otherwise, set x to previous x.
        sum_rates = sum(rates)
        if sum_rates > 0:
            
            # Compute the cdf of the transition probabilities
            cdf = np.cumsum(rates/sum_rates)
        
            #Choose the time of next event
            k   = np.random.uniform(0,1) #Choose a uniform random number
            t  += np.random.exponential(1.0/sum_rates)
        
            # Choose event by finding where k falls within the array cdf.
            which_transition = np.where(k<=cdf)[0][0]
    
            # Add corresponding row of transition matrix to previous state array x,
            # but save previous states in x_prev.
            xt_prev = copy(xt)
            xt += trans_matrix[which_transition]
        
            # if time exceeds next observation reporting time, 
            # return data of previous state
            if t > Treport:
                x.append(list(xt_prev))
            
                # get next reporting time
                j += 1
                if j < len(T):
                    Treport = T[j]
                else:
                    print('I am breaking at A')
                    break
            
            # if time exceeds last reporting time, we're done!
            if t > T[-1]:
                print('I am breaking at B')
                break
                
                
        else:
            print('I am breaking at C')
            break 
            
    return np.array(x)
    


    