# Run Nested Sampling on the model imported from here (default is the exponential varying-hazard model)
include("models/Exponential varying-hazard.jl")

# Set the name of the player you are analysing (e.g. "Oliver Stevenson")
name = "Player name"

## Set the filepath of the data (e.g. "../Data/Stevenson.txt")
data = readdlm(".txt")

# Tuning parameters
num_particles = 1000
mcmc_steps = 1000

# Depth in nats
max_depth = 20.0

# Do an NS run
include("Sampler.jl")
do_nested_sampling(num_particles, mcmc_steps, max_depth)