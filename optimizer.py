# ---------------------------------------------#
# -------| Written By: Sibt ul Hussain |-------#
# ---------------------------------------------#

#        Main class that implements the basic gradient descent 
#        Optimization method, all different flavors of gradient
#        descend  from it.

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


class Optimizer:
    """
        Main class that implements the basic gradient descent 
        Optimization method, all different flavors of gradient
        descend  from it.
    """

    def __init__(self, alpha=0.001, maxniter=2000, plotcf=True):
        """
            Input:
            -----------
            alpha: learning rate set to 0.001
            maxniter: maximum number of iterations of gradient descent
                          to run.
            plotcf: if true, cost function is ploted after every 20 iterations...
        """
        self.alpha = alpha
        self.maxniter = maxniter
        self.plotcf = plotcf

    @staticmethod
    def gradient_check(X, Y, cost_function, derivative_cost_function):
        """
            Function check the gradient of given cost function
            using relative error. i.e. abs(f(x)-f(y))/( abs(fx)+ abs(fy) )
            relative error > 1e-2 usually means the gradient is probably wrong
            1e-2 > relative error > 1e-4 should make you feel uncomfortable
            1e-4 > relative error is usually okay for objectives with kinks.
                   But if there are no kinks (e.g. use of tanh nonlinearities
                   and softmax), then 1e-4 is too high.
            1e-7 and less you should be happy.


            Input:
            -----------
            
            Switch off the regularization before running the gradientCheck
            code...

            X: Input examples (m x d)
            Y: True labels (m x 1)
            cost_function: a function used to evaluated cost
            derivative_cost_function: a function that returns analytical 
                                      derivative of cost function



        """
        thetas = np.random.normal(size=(X.shape[1], 1))
        ad = derivative_cost_function(X, Y, thetas)
        print thetas, ad
        eps = 1e-4
        cd = []

        for i in range(len(thetas)):
            ttpe = thetas.copy()
            ttpe[i] = ttpe[i] + eps  # add an epsilon for the current theta
            ttme = thetas.copy()
            ttme[i] = ttme[i] - eps  # add an epsilon for the current theta
            print ttpe, 'diff in direction', ttpe - ttme
            # import pdb
            # pdb.set_trace()
            fxpe = cost_function(X, Y, ttpe)
            fxme = cost_function(X, Y, ttme)
            # cd.append((cost_function(X,Y,ttpe)-cost_function(X,Y,ttme))/(2*eps))
            cd.append((fxpe - fxme) / (2 * eps))

        print 'Computational derivatvie =', cd
        print 'Analytical derivative =', ad.flatten()
        print 'Relative Error =', np.abs(cd - ad.flatten()) / (np.abs(cd) + np.abs(ad.flatten())+np.spacing(1))

    def gradient_descent(self, X, Y, cost_function, derivative_cost_function):
        '''
            Finds the minimum of given cost function using gradient descent.
            
            Input:
            ------
                X: can be either a single n X d-dimensional vector 
                    or n X d dimensional matrix of inputs            
                
                Y: Must be n X 1-dimensional label vector
                cost_function: a function to be minimized, must return a scalar value
                derivative_cost_function: derivative of cost function w.r.t. paramaters, 
                                           must return partial derivatives w.r.t all d parameters                                                           
                        
            Returns:
            ------
                thetas: a d X 1-dimensional vector of cost function parameters 
                        where minimum point occurs (or location of minimum).
        '''

        # Remember you must plot the cost function after set of iterations to
        # check whether your gradient descent code is working fine or not...
        eps = 0.00001
        # print X.shape
        nexamples = float(X.shape[0])

        # randomly initialize thetas with small random values in the
        # range [-0.003, 0.003]
        thetas = np.random.normal(scale=0.001,size=(X.shape[1], 1))  # np.ones((X.shape[1],1))*0.0001
        # print thetas

        c = 0
        
        cf = []
    
        theta_old = thetas
        theta_new = thetas + 1

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        flag = True
        while c < self.maxniter and flag == True:
            theta_old = theta_new
            theta_new = theta_new - self.alpha * derivative_cost_function(X, Y, np.array(theta_old))
            cf = cost_function(X, Y, theta_new)
            sub = [abs(a-b) for a,b in zip(theta_new, theta_old)]
            sub = [sub > eps]
            if False in sub:
                flag = False
            
            c = c + 1
        
            
        
        #---------End of Your Code-------------------------#

        print 'Value of Cost Function at Minimum Points {}, is {}'.format(theta_new, cf[-1])
        return theta_new
