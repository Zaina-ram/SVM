import numpy , random , math
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class SVM: 
    def __init__(self, N, C, t, data_points):
        self.N = N
        self.C = C
        self.data_points = data_points
        self.start = numpy.zeros(N)
        self.bounds = [(0, C) for b in range(N)]
        self.t = t
        self.P = self.compute_P()
        
    def Minimize(self):
        XC = [{'type' : 'eq', 'fun':self.zerofun}]
        ret = minimize(self.objective, self.start, bounds=self.bounds, constraints=XC)
        alpha = ret['x']
        return alpha

    # Linear kernel function
    # TODO: implement other kernels? 
    def kernel(self, X , Y):
        K = numpy.dot(numpy.transpose(X), Y)
        return K
    
    def polynomial_kernel(self, x, y, degree=10, bias=1):
        """
        Computes the polynomial kernel between two vectors x and y.
        
        Parameters:
        x, y : array-like
            Input vectors.
        degree : int, optional
            Degree of the polynomial kernel function (default is 2).
        bias : int, optional
            Bias term (default is 1).
        
        Returns:
        float
            The polynomial kernel value.
        """
        return (bias + numpy.dot(x, y)) ** degree

    def rbf_kernel(self, x, y, gamma=10):
        """
        Computes the Radial Basis Function (RBF) kernel between two vectors x and y.
        
        Parameters:
        x, y : array-like
            Input vectors.
        gamma : float, optional
            Gamma parameter for the RBF kernel (default is 1.0).
        
        Returns:
        float
            The RBF kernel value.
        """
        # Compute the squared Euclidean distance
        squared_distance = numpy.sum((x - y) ** 2)
        # Compute the RBF kernel value
        kernel_value = numpy.exp(-gamma * squared_distance)
        return kernel_value
    
    # Calculates the objective function in the dual formulation of the SVM. 
    # Uses a pre-computed matrix P
    def objective(self, alpha):
        first_term = numpy.sum(numpy.fromiter((alpha[i] * alpha[j] * self.P[i][j] for i in range(self.N) for j in range(self.N)), dtype=float))
        second_term = numpy.sum(alpha)
        return  (0.5 * first_term - second_term)
    
    # Check contraints for alpha values
    def zerofun(self, alpha):
        dot_product = numpy.dot(alpha, self.t)
        return dot_product

    # extract the non-zero alpha values
    def extract_non_zero(self,alpha):
        alpha_ = []
        t_ = []
        data_points_ = []
        for i in range(self.N):
            if alpha[i] > 1e-5:
                alpha_.append(alpha[i])
                t_.append(self.t[i])
                data_points_.append(self.data_points[i])

        return alpha_, t_, data_points_

    def calculate_b(self, alpha_, t_, data_points_):
        s = 0
        for i in range(len(alpha_)):
            if alpha_[i] > math.pow(10,-5) and alpha_[i] < self.C:
                s = i
                break
        b = numpy.sum(numpy.fromiter((alpha_[i] * t_[i] * self.rbf_kernel(data_points_[s], data_points_[i]) for i in range(len(data_points_))), dtype=float))
        
        return b - t_[s]

    def indicator(self, alpha_, t_, data_points_, b, s):
        ind = numpy.sum(numpy.fromiter((alpha_[i] * t_[i] * self.rbf_kernel(s, data_points_[i]) for i in range(len(alpha_))), dtype=float)) 
        return ind - b


    def compute_P(self):
        P = numpy.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):      
              P[i,j] = self.t[i] * self.t[j] * self.rbf_kernel(self.data_points[i],self.data_points[j])
        return P


   
######################################################SVMinitilization###############################################################
numpy.random.seed(100)

# Linearly seperable
#classA=numpy.random.randn(10,2)*0.2+[1.5,0.5]
#classB=numpy.random.randn(20,2)*0.2+[0.0,-0.5]

# Less complex 
#classA=numpy.concatenate((numpy.random.randn(10,2)*0.2+[1.5,0.5],numpy.random.randn(10,2)*0.2+[-1.5,0.5]))
#classB=numpy.random.randn(20,2)*0.2+[0.0,-0.5]

# Less complex - veryS close - high noise
classA=numpy.concatenate((numpy.random.randn(10,2)*0.2+[0.5,0],numpy.random.randn(10,2)*0.2+[-0.5,0]))
classB=numpy.random.randn(20,2)*0.2+[0.0,-0.5]

# Very complex
#classA=numpy.concatenate((numpy.random.randn(5,2)*0.2+[1.5,0.5],numpy.random.randn(5,2)*0.2+[0,-1.5], numpy.random.randn(5,2)*0.2+[-1.5,0.5], numpy.random.randn(5,2)*0.2+[0,1.5]))
#classB=numpy.concatenate((numpy.random.randn(10,2)*0.2+[0.0,-0.5],numpy.random.randn(10,2)*0.2+[0.0,0.5]))

# Generate very noisy data for 
# classA = numpy.concatenate((numpy.random.randn(10, 2) * 0.5 + [0.5, 0], numpy.random.randn(10, 2) * 0.5 + [-0.5, 0]))
# classB = numpy.random.randn(20, 2) * 0.2 + [0.0, -0.5]

inputs = numpy.concatenate((classA, classB))
targets = numpy.concatenate((numpy.ones(classA.shape[0]), -numpy.ones(classB.shape[0])))
N = inputs.shape[0]  # Number of rows (samples)

# Initialize SVM parameters
C = 0.5
svm = SVM(N, C, targets, inputs)
alpha = svm.Minimize()
print(alpha)
alpha_, t_, data_points_ = svm.extract_non_zero(alpha)
b = svm.calculate_b(alpha_, t_, data_points_)

# Plotting
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
plt.axis('equal')
plt.savefig('svmplot.pdf')

xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)
grid = numpy.array([[svm.indicator(alpha_, t_, data_points_, b, numpy.array([x,y])) for x in xgrid] for y in ygrid])
plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
plt.show()
'''
1. Move the clusters around and change their sizes to make it easier or
harder for the classifier to find a decent boundary. Pay attention
to when the optimizer (minimize function) is not able to find a
solution at all.

2. Implement the two non-linear kernels. You should be able to clas-
sify very hard data sets with these.

3. The non-linear kernels have parameters; explore how they influence
the decision boundary. Reason about this in terms of the bias-
variance trade-off.

poly kernel: 
higher degree: [more flexible decision boundry but also prone to sensitivity to noise and overfitting] 


4. Explore the role of the slack parameter C. What happens for very
large/small values?

5. Imagine that you are given data that is not easily separable. When
should you opt for more slack rather than going for a more complex
model (kernel) and vice versa?
'''

