#This file gives a demo of how the GP package handles non-Gaussian data on a classification example

#using PyPlot, GaussianProcesses
using GaussianProcesses

X = rand(20)
X = sort(X)
y = sin(10*X)
y=convert(Vector{Bool}, y.>0)

#scatter(X,y)

#Select mean, kernel and likelihood function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)
lik = Bernoulli()

gp = GP{Bool}(X',vec(y),mZero,kern,lik)      #Fit the GP, where -1.0 is the log Gaussian noise

optimize!(gp)

