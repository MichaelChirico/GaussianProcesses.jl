# Estimate the parameters of the Gaussian Process using MCMC algorithms

using Gadfly, GaussianProcesses, Lora

srand(13579)

# Training data
n=10                 #number of training points
x = 2π * rand(n)              
y = sin(x) + 0.05*randn(n)

#Select mean and covariance function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

gp = GP(x,y,mZero,kern,-1.0)      #Fit the GP, where -1.0 is the log Gaussian noise

#Estimate the parameters using MCMC

#The default implementation uses the random-walk Metropolis algorithm and details can be found in the mcmc function documentation. Note that the parameters of the MCMC algorithm are automatically tuned.

MCMCout1 = mcmc(gp)

#The user can specify a different sampler from the Lora package as well as different starting values, number of iterations, length of burnin, etc.

#MALA sampler
MCMCout2=mcmc(gp,start=[-1.0,0.0,0.0,0.0],
              sampler=MALA(0.3),
              mcrange=BasicMCRange(nsteps=5000, thinning=2,burnin=1000))

#HMC sampler
MCMCout3=mcmc(gp,start=[-1.0,0.0,0.0,0.0],
              sampler=HMC(5,0.05),
              mcrange=BasicMCRange(nsteps=2000, burnin=1000))

#Plot the marginal posterior for one of the parameters
plot(layer(x=MCMCout1[3,:],Geom.density),
     layer(x=MCMCout2[3,:],Geom.density,Theme(default_color=colorant"red")),
     layer(x=MCMCout3[3,:],Geom.density,Theme(default_color=colorant"green")))



