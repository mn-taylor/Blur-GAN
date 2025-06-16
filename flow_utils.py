import numpy as np

def linear_flow_funcs(num_epochs):
    def t(i):
        working_epochs = num_epochs
        if i<working_epochs:
            return 1 - i/working_epochs
        else: 
            return 0
    
    def e(i):
        return t(i-1) - t(i)
    
    return t, e


def cosine_flow_funcs(num_epochs, warm_up=0, cool_down=0):
    def t(i):
        working_epochs = num_epochs-cool_down-warm_up
        if i-warm_up < working_epochs and i>=warm_up:
            return (1/2) * ( 1 + np.cos(np.pi / (working_epochs) * (i-warm_up)))
        elif i < warm_up:
            return 1
        else:
            return 0
    
    def e(i):
        return np.abs(t(i-1) - t(i))
    
    return t, e

