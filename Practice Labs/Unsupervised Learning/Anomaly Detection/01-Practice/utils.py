import numpy as np

def calc_mean_var(data, axis):        
    mu = []
    sigma2 = []
    
    for i in range(axis):
        mean = data[:, i].mean(axis=0)
        variance = data[:, i].var(axis=0)
        
        mu.append(mean)    
        sigma2.append(variance)
    
    return mu, sigma2


def gaussian_probability(data, mean, variance):
    rows, cols = data.shape
    
    probs = []
    for i in range(cols):
        first_el = 1 / np.sqrt(2 * np.pi * variance[i])
        second_el = np.exp(-((data[:, i] - mean[i]) ** 2) / (2 * variance[i]))
        
        probs.append(first_el * second_el)

    return probs