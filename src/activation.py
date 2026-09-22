import numpy as np

class NonSaturatingActivations:    
    @staticmethod
    def relu(x):
        mask = (x > 0)
        out = np.maximum(0, x)
        
        def backward(dout):
            return dout * mask
            
        return out, backward    
        
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        mask = (x > 0)
        out = np.where(mask, x, alpha * x)
        
        def backward(dout):
            dx = np.where(mask, 1.0, alpha)
            return dout * dx
            
        return out, backward    
        
    @staticmethod
    def elu(x, alpha=1.0):
        mask = (x > 0)
        x_neg = np.clip(x, -80.0, 0.0) 
        exp_part = alpha * (np.exp(x_neg) - 1.0)
        out = np.where(mask, x, exp_part)
        
        def backward(dout):
            dx = np.where(mask, 1.0, out + alpha)
            return dout * dx
            
        return out, backward    
