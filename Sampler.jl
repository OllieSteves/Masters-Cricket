using PyPlot

@doc """
Sampler class
""" ->
type Sampler
	num_particles::Int64
	mcmc_steps::Int64
	particles::Vector{Particle}
	logl::Vector{Float64}
	tiebreakers::Vector{Float64}

	# Current iteration
	iteration::Int64

	# Current log likelihood threshold
	logx_threshold::Float64
	logl_threshold::Float64
	tb_threshold::Float64
end

@doc """
Constructor that only takes num_particles and mcmc_steps
as input
""" ->
function Sampler(num_particles::Int64, mcmc_steps::Int64)
	@assert (num_particles >= 1) & (mcmc_steps >= 1)
	return Sampler(num_particles, mcmc_steps,
								Array(Particle, (num_particles, )),
								zeros(num_particles), zeros(num_particles),
								0, 0.0, -Inf, 0.0)
end

@doc """
Generate all particles from the prior
""" ->
function initialise!(sampler::Sampler)
	for(i in 1:sampler.num_particles)
		sampler.particles[i] = Particle()
		from_prior!(sampler.particles[i])
		sampler.logl[i] = log_likelihood(sampler.particles[i])
		sampler.tiebreakers[i] = rand()
	end
	return nothing
end

@doc """
Find and save worst particle,
then generate replacement.
""" ->
function do_iteration!(sampler::Sampler, verbose::Bool)
	sampler.iteration += 1
	sampler.logx_threshold = -sampler.iteration/sampler.num_particles

	# Find index of worst particle
	worst = find_worst_particle(sampler::Sampler)

	## Get the particles with the lowest likelihood
	worst_particles = sampler.particles[worst]
	C = worst_particles.params[1]
	D = worst_particles.params[3]

	## Need to take the exponential of mu2 as it was modelled log(mu2)
	worst_particles.params[2] = exp(worst_particles.params[2])
	worst_particles.params[1] = worst_particles.params[1] * worst_particles.params[2] # mu1
	worst_particles.params[3] = worst_particles.params[3] * worst_particles.params[2] # L
	
	# Write its information to the output files
	if(sampler.iteration == 1)
		## Save temporary files "sample_info.txt" and "sample.txt"
		## (overwritten with each new player analysed)
		## Save named files "name_info.txt" and "name_"
		## sample_info.txt -
		f = open("sample_info.txt", "w")
		g = open(string(name, "_info.txt"), "w")
		write(f, "# num_particles, iteration, log(X), log(L)\n")
		write(g, "# num_particles, iteration, log(X), log(L)\n")
		## Save as files specified by 'name' in main.jl
		f2 = open("sample.txt", "w")
		g2 = open(string(name, "sample.txt"), "w")
		write(f2, "# The samples themselves. Use log(X) from sample_info.txt as un-normalised prior weights.\n")
		write(g2, "# The samples themselves. Use log(X) from sample_info.txt as un-normalised prior weights.\n")
	else
		f = open("sample_info.txt", "a")
		f2 = open("sample.txt", "a")
		g = open(string(name, "info.txt"), "a")
		g2 = open(string(name, "sample.txt"), "a")
	end
	write(f, string(sampler.num_particles), " ", string(sampler.iteration), " ",
			string(sampler.logx_threshold, " ", string(sampler.logl[worst]), "\n"))
	close(f)
	write(g, string(sampler.num_particles), " ", string(sampler.iteration), " ",
			string(sampler.logx_threshold, " ", string(sampler.logl[worst]), "\n"))
	close(g)
	write(f2, string(string(worst_particles), string(round(C, 5), " "), string(round(D, 5)), "\n")) # Write the worst_particles to sample.txt
	close(f2)
	write(g2, string(string(worst_particles), string(round(C, 5), " "), string(round(D, 5)), "\n"))
	close(g2)

	# Set likelihood threshold
	sampler.logl_threshold = sampler.logl[worst]
	if(verbose)
		println("Iteration ", sampler.iteration, ", log(X) = ",
				sampler.logx_threshold,	", log(L) = ", sampler.logl_threshold)
	end

	# Clone a survivor
	if(sampler.num_particles != 1)
		which = rand(1:sampler.num_particles)
		while(which == worst)
			which = rand(1:sampler.num_particles)
		end
		sampler.particles[worst] = deepcopy(sampler.particles[which])
		sampler.logl[worst] = deepcopy(sampler.logl[which])
		sampler.tiebreakers[worst] = deepcopy(sampler.tiebreakers[which])
	end

	# Evolve the particle
	accepted = 0::Int64
	for(i in 1:sampler.mcmc_steps)
		proposal = deepcopy(sampler.particles[worst])
		logH = perturb!(proposal)
		
		## This just makes sure if we have any values that are infinite, e.g. exp(100000) might give inf, have a numeric value. Exp(0) will give 1 - i.e. a definite acceptance
		if(logH > 0.0)
			logH = 0.0
		end

		logl_proposal = log_likelihood(proposal)
		tb_proposal = sampler.tiebreakers[worst] + randh()
		tb_proposal = mod(tb_proposal, 1.0)

		if((rand() <= exp(logH)) && is_less_than(
						(sampler.logl_threshold, sampler.tb_threshold),
						(logl_proposal, tb_proposal)))
			sampler.particles[worst] = proposal
			sampler.logl[worst] = logl_proposal
			sampler.tiebreakers[worst] = tb_proposal
			accepted += 1
		end
	end
	if(verbose)
		println("Accepted ", accepted, "/", sampler.mcmc_steps, " MCMC steps.\n")
	end

	return (sampler.logx_threshold, sampler.logl_threshold)
end

@doc """
Compare based on likelihoods first. Use tiebreakers to break a tie
""" ->
function is_less_than(x::Tuple{Float64, Float64}, y::Tuple{Float64, Float64})
	if(x[1] < y[1])
		return true
	end
	if((x[1] == y[1]) && (x[2] < y[2]))
		return true
	end
	return false
end


@doc """
Find the index of the worst particle.
""" ->
function find_worst_particle(sampler::Sampler)
	# Find worst particle
	worst = 1
	for(i in 2:sampler.num_particles)
		if(is_less_than((sampler.logl[i], sampler.tiebreakers[i]),
						(sampler.logl[worst], sampler.tiebreakers[worst])))
			worst = i
		end
	end
	return worst
end

@doc """
Calculate the log evidence, information, and posterior weights from the output of a run
""" ->
function calculate_logZ(logX::Vector{Float64}, logL::Vector{Float64})
	# Prior weights
	log_prior = logX - logsumexp(logX)
	# Unnormalised posterior weights
	log_post = log_prior + logL
	# log evidence and information
	logZ = logsumexp(log_post)
    post = exp(log_post - logZ)
	H = sum(post .* (log_post - logZ - log_prior))
	return (logZ, H, log_post)
end

@doc """
Do a Nested Sampling run.
""" ->
function do_nested_sampling(num_particles::Int64, mcmc_steps::Int64,
												depth::Float64; plot=true,
												verbose=true)

	# Number of NS iterations
	steps = Int64(max_depth*mcmc_steps)
	
	# Create the sampler
	sampler = Sampler(num_particles, mcmc_steps)
	initialise!(sampler)

	# Do 'steps' iterations of NS
	# Storage for results
	steps = Int64(max_depth)*num_particles
	plot_skip = num_particles

	# Store logX, logL
	keep = Array(Float64, (steps, 2))

	for(i in 1:steps)
		(keep[i, 1], keep[i, 2]) = do_iteration!(sampler, verbose)
		if(plot && (rem(i, plot_skip) == 0))
			(logZ, H, log_post) = calculate_logZ(keep[1:i, 1], keep[1:i, 2])
			uncertainty = sqrt(H/sampler.num_particles)

			PyPlot.subplot(2, 1, 1)
			PyPlot.hold(false)
			PyPlot.plot(keep[1:i, 1], keep[1:i, 2], "bo-", markersize=1)
			PyPlot.ylabel("\$\\ln(L)\$")
			PyPlot.title(string("\$\\ln(Z) =\$ ", signif(logZ, 6),
						" +- ", signif(uncertainty, 3),
						", \$H = \$", signif(H, 6), " nats"))

			# Adaptive ylim (exclude bottom 5%)
			logl_sorted = sort(keep[1:i, 2])
			lower = logl_sorted[1 + Int64(floor(0.05 * i))]
			PyPlot.ylim([lower, logl_sorted[end] + 0.05 * (logl_sorted[end] - lower)])

			PyPlot.subplot(2, 1, 2)
			PyPlot.plot(keep[1:i], exp(log_post - maximum(log_post)), "bo-", markersize = 1)
			PyPlot.xlabel("\$\\ln(X)\$")
			PyPlot.ylabel("Relative posterior weights")

			PyPlot.draw()
		end
	end

    results = calculate_logZ(keep[:,1], keep[:,2])
    post = exp(results[3] - results[1])
    writedlm("posterior_weights.txt", post)
	writedlm(string(name, "weights.txt"), post)

	if(verbose)
		println("Done!")
	end
	#if(PyPlot.plot)
		#PyPlot.ioff()
		#PyPlot.show()
		#end

end

