import jax.numpy as np


def circ_gauss(x, w):
    # Circular Gaussian from 0 to 180 deg
    return np.exp((np.cos(x * np.pi/90) - 1) / np.square(np.pi/90 * w))


def block_matrix(V, d):
    # Return a matrix of blocks of values
    return np.block([[V[0,0]*np.ones((d[0], d[0])), V[0,1]*np.ones((d[0], d[1]))],
                     [V[1,0]*np.ones((d[1], d[0])), V[1,1]*np.ones((d[1], d[1]))]])


def Euler2fixedpt(dxdt, x_initial, Nmax=100, Navg=10, dt=0.001, xtol=1e-5, xmin=1e-0):
    """
    Finds the fixed point of the D-dim ODE set dx/dt = v(x) (where the function v(.) is called dxdt(.) in this code) 
    using the Euler update with sufficiently large dt (to gain in computational time).
    Checks for convergence to stop the updates early.
    IN:
    dxdt = a function handle giving the right hand side function of dynamical system
    x_initial = initial condition for state variables (a column vector)
    Nmax = maximum iterations to run the Euler
    Navg = number of iterations at the end for which mean step size is taken
    dt = time step of Euler
    xtol = tolerance in relative change in x for determining convergence
    xmin = for x(i)<xmin, it checks convergenece based on absolute change, which must be smaller than xtol*xmin
        Note that one can effectively make the convergence-check purely based on absolute,
        as opposed to relative, change in x, by setting xmin to some very large
        value and inputting a value for 'xtol' equal to xtol_desired/xmin.
               
    OUT:
    xvec = found fixed point solution
    (avg_sum / Navg) = average dx normalised by xtol
    """

    avgStart = Nmax - Navg
    avg_sum = 0
    xvec = x_initial
    
    for n in range(avgStart):  # Loop without taking average step size
        dx = dxdt(xvec) * dt
        xvec = xvec + dx        

    for n in range(Navg):  # Loop whilst recording average step size
        dx = dxdt(xvec) * dt
        xvec = xvec + dx
        
        avg_sum += np.abs(dx /np.maximum(xmin, np.abs(xvec)) ).max() / xtol

    return xvec, avg_sum / Navg


# This is the input-output function (for mean-field spiking neurons) that you would use Max
def Phi(mu, sigma, tau=0.01, Vt=20, Vr=0, tau_ref=0):
    """
     Calculates rate from the Ricciardi equation, with an error
     less than 10^{-5}. If the LIF neuron's voltage, V, satisfies between spikes
                  tau dV/dt = mu - V + sqrt(tau)*sigma*eta(t), 
     where eta is standard white noise, ricciardi calculates the firing rate.
     YA: this is based on Ran Darshan's Matlab version of 
         Carl Van Vreeswijk's Fortran (?) code.
         
     In: 
         mu and sigma: mean and SD of input (in mV)
             can be numpy arrays. If sigma is not scalar, no zero components are allowed.
             If either mu or sigma is scalar, output will have the shape of the other. 
             If both are arrays, both are flattened and output will be of
             shape (mu.size, sigma.size)
         tau, Vt, and Vr:
                 have to be scalar, and denote the membrane time-constant (in seconds), 
                 spiking threshold and reset voltages (in mV), respectively. 
     Out: firing rate (in Hz), with shape:
          if either mu or sigma is scalar, output will have the shape of the other. 
          if both mu and sigma are arrays, both are flattened and 
          rate.shape = (mu.size, sigma.size)
    """

    xp = (mu - Vr) / sigma
    xm = (mu - Vt) / sigma
    

    rate = np.zeros_like(xm)
    
    xm_pos = xm > 0
    rate = (rate * (1 - xm_pos)) + np.nan_to_num(xm_pos / (f_ricci(xp) - f_ricci(xm)))
    
    inds = (xp > 0) & (xm <= 0)
    rate = (rate * (1 - inds)) + np.nan_to_num(inds / ( f_ricci(xp) + np.exp(xm**2) * g_ricci(-xm)))
    
    
    xp_neg = xp <= 0
    rate = (rate * (1 - xp_neg)) + np.nan_to_num(xp_neg * ( np.exp(-xm**2 - np.log(g_ricci(-xm) 
                         - np.exp(xp**2 - xm**2) * g_ricci(-xp)))))
    
    rate = 1 / (tau_ref + 1 / rate)
    
    return rate / tau


def lif_regular(mu, tau, Vt, Vr):
    rate = np.zeros_like(mu)
    rate[mu > Vt] = 1 / np.log((mu[mu > Vt] - Vr)/(mu[mu > Vt] - Vt))

    return rate / tau

def f_ricci(x):
    z = x / (1 + x)
    a = np.array([0.0, 
                  .22757881388024176, .77373949685442023, .32056016125642045, 
                  .32171431660633076, .62718906618071668, .93524391761244940, 
                  1.0616084849547165, .64290613877355551, .14805913578876898])

#    return np.log(2*x + 1) + a @ (-z)**np.arange(10)
    return np.log(2*x + 1) + (  a[1] *(-z)**1 + a[2] *(-z)**2 + a[3] *(-z)**3
                              + a[4] *(-z)**4 + a[5] *(-z)**5 + a[6] *(-z)**6
                              + a[7] *(-z)**7 + a[8] *(-z)**8 + a[9] *(-z)**9 )

def g_ricci(x):

    z = x / (2 + x)
    enum = (  3.5441754117462949 * z    - 7.0529131065835378 * z**2 
            - 56.532378057580381 * z**3 + 279.56761105465944 * z**4 
            - 520.37554849441472 * z**5 + 456.58245777026514 * z**6  
            - 155.73340457809226 * z**7 )
    
    denom = (1 - 4.1357968834226053 * z - 7.2984226138266743 * z**2 
            + 98.656602235468327 * z**3 - 334.20436223415163 * z**4 
            + 601.08633903294185 * z**5 - 599.58577549598340 * z**6 
            + 277.18420330693891 * z**7 - 16.445022798669722 * z**8)
    
    return enum / denom
