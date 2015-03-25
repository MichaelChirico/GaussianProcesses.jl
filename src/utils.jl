# Creates a kernel matrix from vector inputs according to a function d, where D[i,j] = d(x1[i], x2[j]). For example, where d is function of distance between x1 and x2.
#
# Arguments:
#  x1 matrix of observations (each column is an observation)
#  x2 matrix of observations (each column is an observation)
#  d fucntion
function crossKern(x1::Matrix{Float64}, x2::Matrix{Float64}, d::Function)
    dim, nobs1 = size(x1)
    nobs2 = size(x2,2)
    dim == size(x2,1) || throw(ArgumentError("Input observation matrices must have consistent dimensions"))
    dist = Array(Float64, nobs1, nobs2)
    for i in 1:nobs1, j in 1:nobs2
        dist[i,j] = d(x1[:,i], x2[:,j])
    end
    return dist
end

# Returns PD matrix D where D[i,j] = kernel(x1[i], x1[j])
#
# Arguments:
#  x matrix of observations (each column is an observation)
#  d is a function between two vectors
function crossKern(x::Matrix{Float64}, d::Function)
    dim, nobsv = size(x)
    dist = Array(Float64, nobsv, nobsv)
    for i in 1:nobsv
        for j in 1:i
            dist[i,j] = d(x[:,i], x[:,j])
            if i != j; dist[j,i] = dist[i,j]; end;
        end
    end
    return dist
end


# Returns matrix where D[i,j] = kernel(x1[i], x2[j])
#
# Arguments:
#  x1 matrix of observations (each column is an observation)
#  x2 matrix of observations (each column is an observation)
#  k kernel object
function crossKern(x1::Matrix{Float64}, x2::Matrix{Float64}, k::Kernel)
    d(x,y) = kern(k, x, y)
    return crossKern(x1, x2, d)
end

# Returns PD matrix of distances D where D[i,j] = kernel(x1[i], x1[j])
#
# Arguments:
#  x matrix of observations (each column is an observation)
#  k kernel object
function crossKern(x::Matrix{Float64}, k::Kernel)
    d(x,y) = kern(k, x, y)
    return crossKern(x, d)
end

# Calculates the stack [dk / dθᵢ] of kernel matrix gradients
function grad_stack(x::Matrix{Float64}, k::Kernel)
    n = num_params(k)
    d, nobsv = size(x)
    stack = Array(Float64, nobsv, nobsv, n)
    for j in 1:nobsv, i in 1:nobsv
        stack[i,j,:] = grad_kern(k, x[:,i], x[:,j])
    end
    return stack
end