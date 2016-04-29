@doc """
# Description
A function for running a variety of MCMC algorithms for estimating the GP hyperparameters. This function uses the MCMC algorithms provided by the Lora package and the user is referred to this package for further details.

# Arguments:
* `gp::GP`: Predefined Gaussian process type
* `start::Vector{Float64}`: Select a starting value, default is taken as current GP parameters
* `sampler::Lora.MCSampler`: MCMC sampler selected from the Lora package
* `mcrange::Lora.BasicMCRange`: Choose number of MCMC iterations and burnin length, default is nsteps=5000, burnin = 1000
* `noise::Bool`: Noise hyperparameters should be optmized
* `mean::Bool`: Mean function hyperparameters should be optmized
* `kern::Bool`: Kernel function hyperparameters should be optmized
""" ->
function mcmc(gp::GP; start::Vector{Float64}=get_params(gp), sampler::Lora.MCSampler=Lora.MH(ones(length(GaussianProcesses.get_params(gp)))), mcrange::Lora.BasicMCRange=BasicMCRange(nsteps=5000, burnin=1000), noise::Bool=true, mean::Bool=true, kern::Bool=true)
    
    function mll(hyp::Vector{Float64})  #log-target
        set_params!(gp, hyp; noise=noise, mean=mean, kern=kern)
        update_mll!(gp)
        return gp.mLL
    end
    
    function dmll(hyp::Vector{Float64}) #gradient of the log-target
        set_params!(gp, hyp; noise=noise, mean=mean, kern=kern)
        update_mll_and_dmll!(gp; noise=noise, mean=mean, kern=kern)
        return gp.dmLL
    end
    starting = Dict(:p=>start)
    q = BasicContMuvParameter(:p, logtarget=mll,gradlogtarget=dmll) 
    model = likelihood_model(q, false)                               #set-up the model
    tune = VanillaMCTuner(period=mcrange.burnin)                     #set length of tuning (default to burnin length)
    job = BasicMCJob(model, sampler, mcrange, starting,tuner=tune)   #set-up MCMC job
    print(job)                                                       
    run(job)
    chain = Lora.output(job)
    return chain.value
end    
    