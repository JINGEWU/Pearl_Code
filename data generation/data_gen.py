import matplotlib.pyplot as plt
import numpy             as np
import h5py              as h5

from numba           import njit
from scipy.integrate import quad
from tqdm            import tqdm
import sys
# from mpyi4py import MPI
def get_grid(D):
    # Generate distance increments
    dz = np.abs(np.random.randn(D-1)) * 0.01 
    # Compute corresponding grid locations
    z  = np.concatenate(([0.0], np.cumsum(dz)))
    return (z, dz)
def generate_function(M=10):
    # Generate function parameters
    n = np.arange(1,M+1)
    a = np.random.rand(M)
    b = np.random.rand(M)
    c = np.random.rand()
    # Define function
    @njit # Just-in-time compilation to accelerate function evaluation
    def function(x):
        """
        Fourier series with random coefficients a, b, and c.
        """
        y = np.sum(c + np.sin(n*x)*a
                     + np.cos(n*x)*b) / M
        #y = y * np.exp(-.1*x)
        return y
    # Return generated function with random parameters
    return function
def apply(function, array):
    # Generate result vector
    result = np.zeros(array.shape)
    # Apply function to every entry of array
    for i, x in np.ndenumerate(array):
        result[i] = function(x)
    return result
def get_I_bdy():
    return np.random.rand()
def generate_data_point(D=128):
    # Generate grid
    (z, dz) = get_grid(D)
    # Generate emissivity and opacity functions
    eta_func = generate_function(M=15)
    chi_func = generate_function(M=15)
    # Evaluate the function on the grid
    eta = apply(eta_func, z)
    chi = apply(chi_func, z)
    # Get boundary condition
    I_bdy = get_I_bdy()
    # Get helper data to generate target output 
    z_max = np.max(z)    
    def tau(x):
        return quad(chi_func, 0, x)[0]
    def integrand(x):
        return eta_func(x) * np.exp(-tau(x))    
    # Get target output by evaluating the formal solution
    I = I_bdy*np.exp(-tau(z_max)) + quad(integrand, 0, z_max)[0]
    # Return (input, output) pair
    return (z, dz, eta, chi, I_bdy, I)


D = 64
N = int(sys.argv[1])
# N = 20   # number of data points

inputs = np.zeros((N,4*D))
target = np.zeros(N)

for i in tqdm(range(N)):
    (z, dz, eta, chi, Ibdy, I) = generate_data_point(D)
    inputs[i] = np.concatenate((z, dz, eta, chi, [Ibdy]))
    target[i] = I

with h5.File(f'training_data_{N}.hdf5', 'w') as file:
    file['inputs'] = inputs
    file['target'] = target
