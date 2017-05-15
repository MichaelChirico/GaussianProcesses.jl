#This file gives a demo of how the GP package handles non-Gaussian data on a classification example

using Gadfly
using GaussianProcesses

srand(112233)
X = rand(20)
X = sort(X)
y = sin(10*X)
y=convert(Vector{Bool}, y.>0)

plot(x=X,y=y)

#Select mean, kernel and likelihood function
mZero = MeanZero()   #Zero mean function
kern = SE(0.0,0.0)   #Sqaured exponential kernel (note that hyperparameters are on the log scale)
lik = BernLik()

gp = GP(X',vec(y),mZero,kern,lik)     

optimize!(gp)
GaussianProcesses.set_priors!(gp.k,[Distributions.Normal(0.0,2.0),Distributions.Normal(0.0,2.0)])

#mcmc doesn't seem to mix well
samples = mcmc(gp)

plot(y=samples[end,:],Geom.line) #check MCMC mixing

xtest = linspace(minimum(gp.X),maximum(gp.X),50);
ymean = [];
fsamples = [];
for i in 1:size(samples,2)
    GaussianProcesses.set_params!(gp,samples[:,i])
    GaussianProcesses.update_target!(gp)
    push!(ymean, predict_y(gp,xtest)[1])
end



#######################
#Predict 

layers = []
for ym in ymean
    push!(layers, layer(x=xtest,y=ym,Geom.line))
end

plot(layers...,Guide.xlabel("X"),Guide.ylabel("y"))


plot(layer(x=xtest,y=mean(ymean),Geom.line),
     layer(x=X,y=y,Geom.point))

