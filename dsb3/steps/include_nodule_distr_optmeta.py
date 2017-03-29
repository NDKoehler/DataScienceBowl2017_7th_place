import scipy.optimize

def run():
    from include_nodule_distr import run    
    scipy.optimize.bfgs(run, params=params)
    print("optimized parameters")
    
