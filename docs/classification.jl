#This file gives a demo of how the GP package handles non-Gaussian data on a classification example

using PyPlot, GaussianProcesses

X = rand(20,1)
X = sort(X)
y = sin(10*X)
y=y.>0

scatter(X,y)

#Select mean, kernel and likelihood function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)
lik = Bernoulli()

gp = GP(X',vec(y),mZero,kern,lik)      #Fit the GP, where -1.0 is the log Gaussian noise

#     X = np.random.rand(20, 1)
# X = np.sort(X, 0)
# Y = np.sin(10*X)
# Y = np.where(Y>0, 1, 0)
# plt.ion()
# plt.close('all')



