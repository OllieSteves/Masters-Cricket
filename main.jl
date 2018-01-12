# Run Nested Sampling on the model imported from here (default is the exponential varying-hazard model)
include("models/Exponential varying-hazard.jl")

# Set the player name for output files
name = "Fake player"

# Load relevant data file
data = readdlm("Data/Fake data.txt")

# Tuning parameters
num_particles = 1000
mcmc_steps = 1000

# Depth in nats
max_depth = 20.0

# Do a Nested Sampling run
include("Sampler.jl")
do_nested_sampling(num_particles, mcmc_steps, max_depth)
