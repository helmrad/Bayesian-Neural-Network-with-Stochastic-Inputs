# -*- coding: utf-8 -*-
# Created with Python 3.6
"""
Bayesian fully-connected feedforward neural network with stochastic inputs.
Best viewed in the Spyder IDE.
Make sure to:
    - shape training data as [no. of samples, sample-dimension]
    - shape training labels as [no. of samples, target-dimension]
    - specify the input layer size as no. of samples + 2 
      to account for 1 bias and 1 stochastic input
"""

import autograd.numpy as np
from autograd import grad
import pickle
import datetime
import time
import copy
import keyboard
import gc


class BayesianNeuralNetwork:
    
    def __init__(self, layer_sizes):
        
        # Bias and z assumed included in provided layer size
        self.layer_sizes = layer_sizes.copy()
        layer_sizes[-1] = layer_sizes[-1] + 1 # This facilitates weights initialization
        # Initialize weights and their prior variance
        self.w_m = []
        self.w_v = []
        self.w_p_v = 1
        # Initialize random feature z's prior
        self.z_p_v = 2
        # Initialize additive output noise e
        self.e_m = np.zeros((1, self.layer_sizes[-1]))
        self.e_v = np.ones((1, self.layer_sizes[-1]))
        self.activations = []
        
        # Initialize all weights' dimensions (always subtract bias)
        # Initialize all but last layer's activation functions
        for i in range(0, len(layer_sizes)-1):
            w_m = np.zeros((layer_sizes[i], layer_sizes[i+1] - 1)) + np.random.uniform(-0.05,0.05, [layer_sizes[i], layer_sizes[i+1] - 1])
            w_v = np.ones((layer_sizes[i], layer_sizes[i+1] - 1)) * 1e-3
            self.w_m.append(w_m)
            self.w_v.append(w_v)
            self.activations.append(ReLU)
        # Last layer activates trivially
        self.activations[-1] = TrLU
        layer_sizes[-1] = layer_sizes[-1] - 1 # Undo weights initialization facilitation
        
        # Initialize standardization parameters
        self.x_mean = []
        self.x_std  = []
        self.y_mean = []
        self.y_std  = []
        
        # Initialize training parameters
        # Amount of weights, stochastic inputs, and output disturbances to sample
        self.K = 25
        # Alpha
        self.alpha = 0.5
        # Adam's params
        self.step_size = 0.0001
        self.epochs = 10000
    
    
    def standardize(self, x, y = []):
        
        # Extract or use existing normal parameters to normalize samples
        if len(self.x_mean) == 0:
            self.x_mean = np.mean(x, axis = 0)
            self.x_std  = np.std(x, axis = 0)
        x = (x - self.x_mean)/self.x_std
        # Extract or use existing normal parameters to normalize targets if there are any
        if len(y) != 0:
            if len(self.y_mean) == 0:
                self.y_mean = np.mean(y, axis = 0)
                self.y_std  = np.std(y, axis = 0)
            y = (y - self.y_mean)/self.y_std
            return x, y
        else:
            return x
    
    
    def unstandardize(self, x, y):
        
        # Unstandardize samples and targets
        x = x*self.x_std + self.x_mean
        y = y*self.y_std + self.y_mean
        return x, y
    
    
    def sample_NNs(self, w_m, w_v):
        # NNs: deterministic neural networks, zs: random input features, es: additive output noise.
        # Sample K deterministic neural networks with output noise
        NNs = []
        for k in range(0, self.K):
            # Generate weights
            w = []
            for mesh in range(0, len(w_m)):
                rnd = np.random.randn(w_m[mesh].shape[0], w_m[mesh].shape[1])
                w.append(w_m[mesh] + rnd * w_v[mesh])
            NN = DNN(w, self.activations)
            NNs.append(NN)
        return NNs
    
    
    def sample_zs(self, z_m, z_v, N):
        # Sample K sets of N random features
        zs = []
        for k in range(0, self.K):
            rnd = np.random.randn(N, z_m.shape[1])
            z = z_m + rnd * z_v
            zs.append(z)
        return zs
    
    
    def sample_es(self, e_v, N):
        # Generate additive output noise for K NNs and N NN-outputs
        rnd = np.random.randn(self.K, N, self.e_m.shape[1])
        es = self.e_m + rnd * e_v
        return es
    
    
    def compute_norm_factor(self, w_m, w_v, z_m, z_v):
        
        # Calculate normalization factor for the weights
        log_n_w = 0
        for mesh in range(0, len(w_m)):
            log_n_w = log_n_w + np.sum(0.5 * np.log(2*np.pi*w_v[mesh]) + np.square(w_m[mesh])/w_v[mesh])
        # And for the random features
        log_n_z = np.sum(0.5 * np.log(2*np.pi*z_v) + np.square(z_m)/z_v)
        # Assemble total normalization factor
        log_n = log_n_w + log_n_z
        return log_n
    
    
    def compute_LL_factors(self, NNs, zs, w_m, w_v, z_m, z_v, N):
        
        # Calculate K likelihood factors for the weights
        f_ws = []
        f_zs = []
        for k in range(0, self.K):
            # For the weights
            f_w = 0
            for mesh in range(0, len(NNs[k].w)):
                # Split long expression into two
                f1_w = ((self.w_p_v*w_v[mesh])/(self.w_p_v-w_v[mesh])) * np.square(NNs[k].w[mesh]) 
                f2_w = (w_m[mesh] / w_v[mesh]) * NNs[k].w[mesh]
                f_w  = f_w + (np.sum(f1_w) + np.sum(f2_w))/N
            f_w = f_w * self.alpha
            f_w = np.exp(f_w)
            if f_w > 1e150:
                print('Warning: an f_w converges to inf.')
                self.errormsg.append('an f_w goes to inf.')
            f_ws.append(f_w)
            # For the random features z
            f1_z = ((self.z_p_v*z_v)/(self.z_p_v-z_v)) * np.square(zs[k])
            f2_z = (z_m/z_v) * zs[k]
            f_z = (f1_z + f2_z)*self.alpha
            f_z = np.exp(f_z)
            if (f_z > 1e150).any() == True:
                print('Warning: an f_z converges to inf.')
                self.errormsg.append('an f_z goes to inf.')
            f_zs.append(f_z)
        return f_ws, f_zs
    
    
    def compute_sample_LLs(self, NNs, zs, w_m, w_v, z_m, z_v, e_v, x, y, N):
        
        # Compute likelihood factors 
        f_ws, f_zs = self.compute_LL_factors(NNs, zs, w_m, w_v, z_m, z_v, N)
        # Calculate likelihoods for every data-pair in the batch
        lls = []
        denom = 2 * e_v
        for k in range(0, self.K):
            # Append random features z to x
            x_z = np.concatenate((x, zs[k]), axis = 1)
            out = NNs[k].execute(x_z)
            nom = np.square(y-out)
            ll = np.exp(-nom/denom) / (np.sqrt(2 * np.pi * e_v)) + 1e-10
            # Multiply multi-dimensional output if applicable
            ll = np.prod(ll, axis=1, keepdims=True)
            if (ll == 0).any() == True:
                print('Warning: A likelihood is zero.', np.argmin(ll))
                self.errormsg.append('one ll is zero.', np.argmin(ll))
            # Include alpha and divide by likelihood factors
            factored_ll = (ll**self.alpha / (f_ws[k]*f_zs[k]))
            if (factored_ll == 0).any() == True:
                print('Warning: A factored likelihood is zero.')
                self.errormsg.append('one f_ll is zero.')
            lls.append(factored_ll)
        return lls
    
    
    def compute_BNN_LL(self, lls, N):
        
        # Compute the likelihood of the BNN parameters (second part energy function)
        ll_sums_per_sample = np.zeros((N, lls[0].shape[1]))
        for k in range(0, self.K):
            ll_sums_per_sample = ll_sums_per_sample + lls[k]
        log_ll_sums_per_sample = np.log(ll_sums_per_sample/self.K)
        ll_BNN = np.sum(log_ll_sums_per_sample)/self.alpha
        return ll_BNN
    
    
    def calculate_energy(self, tbo_pars, x, y, N):
        
        # Unpack parameters
        w_m = tbo_pars[0]
        w_v = tbo_pars[1]
        z_m = tbo_pars[2]
        z_v = tbo_pars[3]
        e_v = tbo_pars[4]
        # Generate deterministic neural networks and feature noise
        zs  = self.sample_zs(z_m, z_v, N)
        NNs = self.sample_NNs(w_m, w_v)
        # Compute normalization factor of the approximating distribution
        log_n = self.compute_norm_factor(w_m, w_v, z_m, z_v)
        # Compute likelihoods for all data-pairs per sampled NN
        lls = self.compute_sample_LLs(NNs, zs, w_m, w_v, z_m, z_v, e_v, x, y, N)
        # Compute the average normalized likelihood of the BNN's parameters
        ll_BNN = self.compute_BNN_LL(lls, N)
        # Calculate the energy value
        energy = - log_n - ll_BNN
        return energy
    
    
    def adam_deluxe(self, tbo_pars, b1=0.9, b2=0.999, eps=10**-8):
        
        # Initialize m and v for every parameter
        m = []
        v = []
        tbo_pars_iter = []
        for par in tbo_pars:
            if type(par) == list:
                m_par = []
                v_par = []
                for mesh in par:
                    m_par.append(np.zeros((mesh.shape[0], mesh.shape[1])))
                    v_par.append(np.zeros((mesh.shape[0], mesh.shape[1])))
            else:
                m_par = np.zeros((par.shape[0], par.shape[1]))
                v_par = np.zeros((par.shape[0], par.shape[1]))
            m.append(m_par)
            v.append(v_par)
        
        print("Training starts. Minimizing the energy function for", self.epochs, "epochs.")
        print("Hold ctrl+shift+p to pause and ctrl+shift+s to stop training. \n")
        # Iterate updates
        for i in range(self.epochs):
            
            # Stop or pause training by pressing specified key
            if keyboard.is_pressed('ctrl+shift+s') == True:
                return tbo_pars, tbo_pars_iter
            elif keyboard.is_pressed('ctrl+shift+p') == True:
                input("Enter sth to resume training")
            
            # Save current params, calculate energy and stop training once energy is nan
            prior_pars = copy.deepcopy(tbo_pars)
            tbo_pars_iter.append(prior_pars)
            e = np.round(self.energy(tbo_pars), 4)
            print(i, 'Energy: ', e)
            if np.isnan(e):
                return tbo_pars, tbo_pars_iter
            # Get the gradients wrt parameters to-be-optimized
            grads = self.energy_grad(tbo_pars)
            # Optimize parameters one by one
            for par_n in range(0, len(tbo_pars)):
                if type(tbo_pars[par_n]) == list:
                    # Optimize weights mesh by mesh
                    for mesh in range(0, len(tbo_pars[par_n])):
                        m[par_n][mesh] = (1 - b1) * grads[par_n][mesh]      + b1 * m[par_n][mesh]
                        v[par_n][mesh] = (1 - b2) * (grads[par_n][mesh]**2) + b2 * v[par_n][mesh]
                        mhat           = m[par_n][mesh] / (1 - b1**(i + 1)) 
                        vhat           = v[par_n][mesh] / (1 - b2**(i + 1))
                        tbo_pars[par_n][mesh] = tbo_pars[par_n][mesh] - self.step_size*mhat/(np.sqrt(vhat) + eps)
                else:
                    m[par_n] = (1 - b1) * grads[par_n]      + b1 * m[par_n]   # First  moment estimate
                    v[par_n] = (1 - b2) * (grads[par_n]**2) + b2 * v[par_n]   # Second moment estimate
                    mhat     = m[par_n] / (1 - b1**(i + 1))                   # Bias correction
                    vhat     = v[par_n] / (1 - b2**(i + 1))
                    tbo_pars[par_n] = tbo_pars[par_n] - self.step_size*mhat/(np.sqrt(vhat) + eps)
            
            # Show statistics
            if i%10 == 0:
                print('MSE:', np.round(self.MSE(), 6))
                adam_training_statistics(prior_pars, tbo_pars)
                
            # Collect garbage
            gc.collect()
            
        return tbo_pars, tbo_pars_iter
    
    
    def train(self, x, y):
        
        self.errormsg = []
        N = x.shape[0]
        # Initialize array of random feature means and variances to fit
        z_m = np.zeros((x.shape[0],1))
        z_v = np.ones((x.shape[0],1)) + np.random.uniform(-0.1,0.1, [x.shape[0], 1])
        # Declare MSE
        self.MSE = lambda: np.mean(np.square(y - np.mean(self.execute(x), axis = 0)))
        
        # Optimize parameters by minimizing the black-box alpha-divergence energy function
        # Declare energy function
        self.energy = lambda tbo_pars: self.calculate_energy(tbo_pars, x, y, N)
        # Declare gradient of energy wrt parameters to-be-optimized
        self.energy_grad = grad(self.energy, 0)
        # Wrap to-be-optimized parameters and optimize
        tbo_pars = [self.w_m, self.w_v, z_m, z_v, self.e_v]
        tbo_pars, tbo_pars_iter = self.adam_deluxe(tbo_pars)
        # Reassign parameters
        self.w_m = tbo_pars[0]
        self.w_v = tbo_pars[1]
        self.e_v = tbo_pars[4]
        # Return course of parameter optimization
        return tbo_pars_iter
    
    
    def execute(self, x):
        
        N = x.shape[0]
        # Generate K x N random features z
        z_m = np.zeros((N, 1))
        z_v = np.ones((N, 1)) * self.z_p_v
        zs = self.sample_zs(z_m, z_v, N)
        # Sample K deterministic neural networks
        NNs = self.sample_NNs(self.w_m, self.w_v)
        # Sample K x N times additive noise e
        es = self.sample_es(self.e_v, N)
        
        # Initialize output array
        out = np.zeros((self.K, N, self.layer_sizes[-1]))
        # Get perturbed outputs of K NNs
        for k in range(0, self.K):
            # Append features z to x, execute NNs, perturb with e
            x_z = np.concatenate((x, zs[k]), axis = 1)
            out[k,:,:] = NNs[k].execute(x_z) + es[k,:,:]
        return out
    
    
    def save(self, note = 0):
        
        # Wrap NN parameters
        NN_params = []
        NN_params.append(self.layer_sizes)
        NN_params.append(self.activations)
        NN_params.append(self.z_p_v)
        NN_params.append(self.w_m)
        NN_params.append(self.w_v)
        NN_params.append(self.e_v)
        NN_params.append(self.x_mean)
        NN_params.append(self.x_std)
        NN_params.append(self.y_mean)
        NN_params.append(self.y_std)
        
        # Save them
        present = datetime.date.today().strftime('%Y%m%d')
        if note == 0:
            name = '_BNN_params.pkl'
        else:
            name = '_BNN_params_' + note + '.pkl'
        with open(present + name, 'wb') as fhandle:
            pickle.dump(NN_params, fhandle)



#%%

# Activation functions

# Rectified Linear Unit
def ReLU(o):
    out = np.maximum(o,0)
    return out

# Trivial Linear Unit
def TrLU(o):
    return o

def tanh(o):
    out = np.tanh(o)
    return out


# Deterministic neural network
class DNN:
    
    def __init__(self, w, acts):
        # Deterministic neural network with scalar weights
        self.w = w
        self.activations = acts
    
    def execute(self, x):
        # Append a bias to every input
        biasses = np.ones((x.shape[0], 1))
        layer_input = np.concatenate((x, biasses), axis = 1)
        # For every computing layer
        for layer in range(0, len(self.w)):
            # Compute the layer's output as its neurons' activated sums of weighted inputs
            layer_output = self.activations[layer](np.dot(layer_input, self.w[layer]))
            # Assign the biassed output as input to the next layer
            layer_input = np.concatenate((layer_output, biasses), axis = 1)
        # Input passed through the entire network yields the prediction y
        out = layer_output
        return out


# Load BNN
def load_BNN(fname):
    with open(fname, 'rb') as fhandle:
        NN_params = pickle.load(fhandle)
        BNNET = BayesianNeuralNetwork(NN_params[0])
        BNNET.activations   = NN_params[1]
        BNNET.z_p_v         = NN_params[2]
        BNNET.w_m           = NN_params[3]
        BNNET.w_v           = NN_params[4]
        BNNET.e_v           = NN_params[5]
        BNNET.x_mean        = NN_params[6]
        BNNET.x_std         = NN_params[7]
        BNNET.y_mean        = NN_params[8]
        BNNET.y_std         = NN_params[9]
    return BNNET


def adam_training_statistics(prior_pars, tbo_pars):
    
    delta = []
    gain = 100000
    for par_bef, par_now in zip(prior_pars, tbo_pars):
        if type(par_bef) == list:
            d = 0
            for mesh in range(0, len(par_bef)):
                d = d + np.mean(np.abs(par_bef[mesh] - par_now[mesh]))
            d = d/len(par_bef)
        else:
            d = np.mean(np.abs(par_bef - par_now))
        delta.append(np.round(d*gain, 2))
    print('AVG update x', gain, 'per parameterset: ', delta[0], delta[1], delta[2], delta[3], delta[4])


#%%
# Application example


def fit_bimodal_distribution():
    
    ''' Fit a BNN on a bimodal process '''
    res     = 200
    base    = np.linspace(-2,2,res)
    db      = np.append(base,base)
    x       = db
    curve_1 = 10*np.sin(base) + np.random.randn(res)
    curve_2 = 10*np.cos(base) + np.random.randn(res)
    y       = np.append(curve_1, curve_2)
    for i in range(0,10):
        base    = np.linspace(-2,2,res + np.random.randint(-30,30))
        curve_1 = 10*np.sin(base) + np.random.randn(len(base))
        curve_2 = 10*np.cos(base) + np.random.randn(len(base))
        db      = np.append(base,base)
        x       = np.append(x,db)
        y_      = np.append(curve_1, curve_2)
        y       = np.append(y, y_)
    x_o = x.reshape(x.shape[0],1)
    y_o = y.reshape(y.shape[0],1)
    del res, base, db, curve_1, curve_2, y_, i, x, y
    
    # Visualize
    import matplotlib.pyplot as plt
    plt.figure(figsize = (18.6, 12.5))
    plt.scatter(x_o, y_o, s = 20, c = 'r', marker = 'x', label = 'Training data points')
    plt.legend(fontsize = 20, loc = 'upper left')
    plt.tick_params(axis='both', which='major', labelsize = 20)
    plt.show()
    plt.pause(5)
    plt.close()
    
    # Setup, preprocess
    # Input layer size +2, for one bias and one random feature
    layer_sizes = [x_o.shape[1]+2, 10, 10, y_o.shape[1]]
    net = BayesianNeuralNetwork(layer_sizes)
    net.step_size = 0.001
    net.epochs = 5000
    x, y = net.standardize(x_o, y_o)
    
    # Train
    _ = net.train(x, y)
    # Execute
    out = net.execute(x)
    # Unstandardize the network output
    _, out_r = net.unstandardize(x, out)
    
    # See the outcome
    import matplotlib.pyplot as plt
    plt.figure(1, figsize = (18.6, 12.5))
    plt.scatter(x_o, y_o, s = 20, c = '#FF2121', marker = 'x', label = 'True distribution')
    for k in range(0, 1):
        plt.scatter(x_o, out_r[k,:,:], s = 15, c = '#4169E1', marker = 'o', alpha = 0.4, label = 'Learned distribution')
    plt.legend(fontsize = 20, loc = 'upper left')
    plt.tick_params(axis='both', which='major', labelsize = 20)
    
    
if __name__ == "__main__":
    print("Hi!")
    print("Example application: fit a Bayesian neural network to a bimodal distribution.")
    time.sleep(3)
    print("See the plot that pops up. \n")
    fit_bimodal_distribution()

