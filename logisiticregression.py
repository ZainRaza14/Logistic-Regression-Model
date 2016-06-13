#---------------------------------------------#
#-------| Written By: Sibt ul Hussain |-------#
#---------------------------------------------#

# A Logistic Regression algorithm with regularized weights...

from classifier import *
import math

#Note: Here the bias term is considered as the last added feature 



class LogisticRegression(Classifier):
    ''' Implements the LogisticRegression For Classification... '''
    def __init__(self, lembda=0.001):        
        """
            lembda= Regularization parameter...            
        """
        self.theta=[] 
        Classifier.__init__(self,lembda)                
        
        pass
    def sigmoid(self,z):
        """
            Compute the sigmoid function 
            Input:
                z can be a scalar or a matrix
            Returns:
                sigmoid of the input variable z
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
            
        
        #---------End of Your Code-------------------------#
    
    
    def hypothesis(self, X,theta):
        '''
            Computes the hypothesis for over given input examples (X) and parameters (thetas).

            Input:
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix
                theta: Must be a d-dimensional vector
            Return:
                The computed hypothesis
        '''
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        return (1 / (1 + math.exp(-1 * sum(value * weight for value, weight in zip(X, theta)))))    
        
        #---------End of Your Code-------------------------#
        
    
        return self.sigmoid(h)
    def cost_function(self, X,Y, theta):
        '''
            Computes the Cost function for given input data (X) and labels (Y).

            Input:
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix of inputs
                theata: must  d X 1-dimensional vector for representing vectors
                Y: Must be n X 1-dimensional label vector with values 1 & 0
                
            Return:
                Returns the cost of hypothesis with input parameters 
        '''
    
    
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        m = X.shape[0]
        cost = sum([(-1 * Y[index] * math.log(self.hypothesis(X[index], theta))) - ((1 + (-1 * Y[index])) * math.log(1 -self.hypothesis(X[index], theta))) for index in range(len(X))]) / m
        
            
        
        #---------End of Your Code-------------------------#
        
        return cost
    def derivative_cost_function(self,X,Y,theta):
        '''
            Computes the derivates of Cost function w.r.t input parameters (thetas)  
            for given input and labels.

            Input:
            ------
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix of inputs
                theata: must  d X 1-dimensional vector for representing vectors
                Y: Must be n X 1-dimensional label vector
            Returns:
            ------
                partial_thetas: a d X 1-dimensional vector of partial derivatives of cost function w.r.t parameters..
        '''
        nexamples=float(X.shape[0])

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        nfeatures = X.shape[1]

        partialderivatives = np.zeros(shape=(nfeatures, 1))

        pre_cal = []
        for index in range(len(X)):
            pre_cal.append(self.hypothesis(X[index], theta) - Y[index])

        for j in range(nfeatures):
            partialderivatives[j] = float(sum([(pre_cal[index] * X[index][j]) for index in range(len(X))]))
        partialderivatives /= nexamples
            
        
        #---------End of Your Code-------------------------#
        return partialderivatives

    def train(self, X, Y, optimizer):
        ''' Train classifier using the given 
            X [m x d] data matrix and Y labels matrix
            
            Input:
            ------
            X: [m x d] a data matrix of m d-dimensional examples.
            Y: [m x 1] a label vector.
            optimizer: an object of Optimizer class, used to find
                       find best set of parameters by calling its
                       gradient descent method...
            Returns:
            -----------
            Nothing
            '''
        
        nexamples,nfeatures=X.shape
        ## now go and train a model for each class...     
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        #if self.scalefeatures:
        #    X=self.scale_features(X)
            
        self.theta = optimizer.gradient_descent(X, Y, self.cost_function, self.derivative_cost_function)
            
        
        #---------End of Your Code-------------------------#
        
        
    
    def predict(self, X):
        
        """
        Test the trained perceptron classifier result on the given examples X
        
                   
            Input:
            ------
            X: [m x d] a matrix of m  d-dimensional test examples.
           
            Returns:
            -----------
                pclass: the predicted class for the given set of examples, i.e. to which it belongs
        """
        
        num_test = X.shape[0]
        
        
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
       
        pclass = np.ones(shape=(num_test, 1))
        
        for index, test in enumerate(X):
            if sum(tval * t for tval, t in zip(test, self.theta)) < 0:
                pclass[index] = 0

        return pclass
            
        
        #---------End of Your Code-------------------------#

        #return Ypred
