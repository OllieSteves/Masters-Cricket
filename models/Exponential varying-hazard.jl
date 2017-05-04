include("../Utils.jl")

using Distributions

@doc """
An object of this class represents a point in parameter space.
There are functions defined to evaluate the log likelihood and
move around.
""" ->
type Particle
	params::Vector{Float64}
end

@doc """
A constructor. We have 6 parameters:
The usual mu_1, mu_2 and the time scale L
As well as the Gaussian parameters k, sigma and m (strength, width and midpoint of Gaussian)
""" ->
function Particle()
	return Particle(Array(Float64, (6, )))
end

@doc """
Generate params from the prior
""" ->
function from_prior!(particle::Particle)
	## Lognormal prior for mu_2
	particle.params[2] = rand(Normal(log(25), 0.75), 1)[1]
	
	##### BETA PRIORS #####
	## Beta priors for C & D
	particle.params[1] = rand(Beta(1, 2), 1)[1]
	particle.params[3] = rand(Beta(1, 5), 1)[1]

	return nothing
end

## Since we are now modelling using 3 parameters we must update the perturb! function
@doc """
Do a metropolis proposal. Return log(hastings factor for prior sampling)
""" ->
function perturb!(particle::Particle)
	## Define logH, length of data (i.e. number of innings)
	logH = 0.0
	innings = length(data[:, 1])
	
	## Randomly decide which parameter we are going to play around with
	i = rand(1:length(particle.params))
	
	## Alter C, logH = p' - p (0 in the case of Uniform priors)
	if(i == 1)
		## Beta prior ##
		a = 1
		b = 2
		
		## Log-prior before
		logH -= (a - 1) * log(particle.params[i]) + (b - 1) * log(1 - particle.params[i])

		## Explore parameter space - map it onto the [0, 1] interval
		particle.params[i] += 1*randh()
		particle.params[i] = mod(particle.params[i], 1)
		
		## Log-prior after
		logH += (a - 1) * log(particle.params[i]) + (b - 1) * log(1 - particle.params[i])
	end
	
	## Alter mu_2, logH = p' - p (0 in the case of Uniform priors)
	if(i == 2)
		## Lognormal prior - technically log(normal) ##
		mu = 3.25
		sig = 0.75
		
		## Log(normal)-prior before
		logH -= -0.5 * log(2 * pi) - log(sig) - (1/(2 * sig^2)) * (particle.params[i] - mu)^2
		
		## Explore the parameter space (use a scale of sigma)
		particle.params[i] = particle.params[i] + sig * randh()
		
		## Log(normal)-prior after
		logH += -0.5 * log(2 * pi) - log(sig) - (1/(2 * sig^2)) * (particle.params[i] - mu)^2
	end
	
	## Alter D, logH = p' - p
	if(i == 3)
		## Beta prior ##
		a = 1
		b = 5
		
		## Log-prior before
		logH -= (a - 1) * log(particle.params[i]) + (b - 1) * log(1 - particle.params[i])

		## Explore parameter space - map it onto the [0, 1] interval
		particle.params[i] += 1 * randh()
		particle.params[i] = mod(particle.params[i], 1)
		
		## Log-prior after
		logH += (a - 1) * log(particle.params[i]) + (b - 1) * log(1 - particle.params[i])
	end
	
	return logH
end


@doc """
The 'effective average' function
"""->
## Using mu_1 = C * mu_2, L = D * mu_2
function effective_average(particle::Particle, data::Array{Float64} = data)
	## Take the exponential of particle parameters mu_1 and mu_2 as they come from a lognormal but have been modelled as normal
	mu2 = exp(particle.params[2])
	mu1 = particle.params[1] * mu2
	L = particle.params[3] * mu2
	return((mu2 + (mu1 - mu2) * exp(-data[:, 1] / L)))
end


@doc """
Evaluate the log likelihood
""" ->
function log_likelihood(particle::Particle, data::Array{Float64, 2} = data)
	logL1 = 0.0
	logL2 = 0.0
	
	## Vector of scores from 0 to batsman's maximum score
	scores = Float64[0:maximum(data[:, 1]); ]
	## In terms of H(x)
	cumsum_scores = cumsum(log(effective_average(particle, scores)) - log(effective_average(particle, scores) + 1))
	
	## Take the x_i - 1 cumulative sum for each score
	for(i in 1:length(data[:, 1]))
		score = Int64[data[i, 1]][1] # convert to integer for indexing (cannot index using Float64)
        if(score >= 1)
    		logL1 += cumsum_scores[score]
        end
	end
	
	## Out scores
	out = data[data[:, 2] .== 0, :]
	logL2 = Float64[sum(-log(effective_average(particle, out[:, 1]) + 1))]
	
	return(logL1[1] + logL2[1])
	
	## Set return = 0.0 to make sure that the perturb function is sampling from the prior
	#return(0.0)
end

@doc """
Convert to string, for output to sample.txt
"""
import Base.string
function string(particle::Particle)
	return join([string(signif(x, 6), " ") for(x in particle.params)])
end