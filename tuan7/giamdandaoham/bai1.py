import numpy as np
from matplotlib import pyplot as plt

def grad(x):
    return 2*x+ 5*np.cos(x)

def cost(x):
    return x**2 + 5*np.sin(x)

def myGD1(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3: 
            break
        x.append(x_new)
    return (x, it)

def main():
    (x1, it1) = myGD1(-5, .1)
    print('x = %4f,cost = %.4f va so lan lap = %d' % (x1[-1], cost(x1[-1]), it1)) 
    #(x1, it1) = myGD1(5, .1)
    #print('x = %4f,cost = %.4f va so lan lap = %d' % (x1[-1], cost(x1[-1]), it1))
    
    x = np.linspace(-6, 6, 100)
    y = x**2 + 5*np.sin(x)
    
    k = 0
    plt.subplot(2,4,1)
    plt.plot(x, y, 'b')
    plt.plot(x1[k], cost(x1[k]), 'ro')
    s = 'iter %d/%d, grad = %.4f' % (k, it1, grad(x1[k]))
    plt.xlabel(s, fontsize=8)
    
    k = 1
    plt.subplot(2,4,2)
    plt.plot(x, y, 'b')
    plt.plot(x1[k], cost(x1[k]), 'ro')
    s = 'iter %d/%d, grad = %.4f' % (k, it1, grad(x1[k]))
    plt.xlabel(s, fontsize=8) 
      
    k = 2
    plt.subplot(2,4,3)
    plt.plot(x, y, 'b')
    plt.plot(x1[k], cost(x1[k]), 'ro')
    s = 'iter %d/%d, grad = %.4f' % (k, it1, grad(x1[k]))
    plt.xlabel(s, fontsize=8)
    
    k = 3
    plt.subplot(2,4,4)
    plt.plot(x, y, 'b')
    plt.plot(x1[k], cost(x1[k]), 'ro')
    s = 'iter %d/%d, grad = %.4f' % (k, it1, grad(x1[k]))
    plt.xlabel(s, fontsize=8)
    
    k = 4
    plt.subplot(2,4,5)
    plt.plot(x, y, 'b')
    plt.plot(x1[k], cost(x1[k]), 'ro')
    s = 'iter %d/%d, grad = %.4f' % (k, it1, grad(x1[k]))
    plt.xlabel(s, fontsize=8)
    
    k = 5
    plt.subplot(2,4,6)
    plt.plot(x, y, 'b')
    plt.plot(x1[k], cost(x1[k]), 'ro')
    s = 'iter %d/%d, grad = %.4f' % (k, it1, grad(x1[k]))
    plt.xlabel(s, fontsize=8)
    
    k = 7
    plt.subplot(2,4,7)
    plt.plot(x, y, 'b')
    plt.plot(x1[k], cost(x1[k]), 'ro')
    s = 'iter %d/%d, grad = %.4f' % (k, it1, grad(x1[k]))
    plt.xlabel(s, fontsize=8)
    
    k = 11
    plt.subplot(2,4,8)
    plt.plot(x, y, 'b')
    plt.plot(x1[k], cost(x1[k]), 'ro')
    s = 'iter %d/%d, grad = %.4f' % (k, it1, grad(x1[k]))
    plt.xlabel(s, fontsize=8)
 
 
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    main()