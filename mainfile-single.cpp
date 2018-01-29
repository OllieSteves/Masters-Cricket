// Exponential Varying Hazard Model - definition of model

// Include relevant files
#include "RNG.h"
#include "sampler.h"
#include "model.h"
#include "getfile.h"

// Packages
#include <iostream>
#include <fstream>
#include <sstream> // For stringstream
#include <vector>
#include <ctime>
#include <string>

int main()
{
	// Enter the name of the player to analyse
	std::string name = "Fake player";
	
	// Get the appropriate player data
	std::string filename = "./Data/";
	filename.append(name);
	filename.append(".txt");

	// Random number generating object
	RNG rng;
	rng.set_seed(time(0));

	// Data
	std::vector<std::vector<int> > data = read_file(filename);	

	// Tuning parameters
	int nparticles = 1000;
	int mcmc_steps = 1000;

	// Depth in nats
	double max_depth = 20.0;

	// Initialise the sampler
	Sampler<Particle> s(data, nparticles, mcmc_steps, name);

	// Do a nested sampling run
	do_nested_sampling(data, nparticles, mcmc_steps, max_depth, name, rng);

	return 0;
}
