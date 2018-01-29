// Nested sampling sampler

#include "sampler.h"
#include "model.h"
#include "Utils.h"

// Packages
#include <iostream>
#include <fstream> // for std::ofstream for writing to file

// Forward declare template
template class Sampler<Particle>;

// Define the Sampler Template class functions
// Sampler - constructor function
// Only takes a number of particles and mcmc steps
// Initiates empty vectors of particles and various log-likelihood thresholds
template <class T>
Sampler<T>::Sampler(std::vector<std::vector<int> > data, int nparticles, int mcmc_steps, std::string name) : 
	data(data), nparticles(nparticles), mcmc_steps(mcmc_steps), name(name),
	particles(std::vector<T> (nparticles)),
	logL(std::vector<double> (nparticles)),
	tiebreakers(std::vector<double> (nparticles))
{
	// Assign initial values
	iteration = 0;
	logX_threshold = 0.0;
	logL_threshold = 0.0;
	tb_threshold = 0.0;
}

// Sampler - initialise function
template <class T>
void Sampler<T>::initialise(RNG& rng)
{
	// Initialise the particles from the prior
	for (int i = 0; i < nparticles; i++)
	{
		particles[i].from_prior(rng); // Generate particle values from the prior
		logL[i] = particles[i].log_likelihood(data); // Calculate the log-likelihood values for each particle
		tiebreakers[i] = rng.rand();
	}
}

// Sampler - do_iteration - an iteration of nested sampling
template <class T>
std::vector<double> Sampler<T>::do_iteration(RNG& rng)
{	
	// Update current iteration and X threshold
	iteration += 1;
	double logX_threshold = static_cast<double>(-iteration)/nparticles;

	// Find index of worst particle in terms of likelihood
	int worst_index = find_worst_particle();

	// Get the worst particle and extract mu1, mu2 and L
	Particle worst = particles[worst_index];
	//double mu2 = exp(worst.params[1]);
	//double mu1 = worst.params[0] * mu2;
	//double L = worst.params[2] * mu2;

	//std::cout << "Worst Index = " << worst_index << ". Worst particle = " << particles[worst_index] << ", Log-likelihood = " << logL[worst_index] << std::endl;

	// Write the worst particle's information to file
	if(iteration == 1)
	{
		// Output file for samples
		std::ofstream outf_sample("sample.txt");
		std::ofstream save_sample(std::string("./Results/") + name + std::string("_sample.txt"));
		outf_sample << worst << std::endl;
		save_sample << worst << std::endl;
				
		// Output file for info
		std::ofstream outf_info("info.txt");
		std::ofstream save_info(std::string("./Results/") + name + std::string("_info.txt"));
		outf_info << nparticles << " " << iteration << "," << logX_threshold << "," << logL[worst_index] << std::endl;
		save_info << nparticles << " " << iteration << "," << logX_threshold << "," << logL[worst_index] << std::endl;
	}
	else
	{
		// Samples
		std::ofstream outf_sample;
		std::ofstream save_sample;
		outf_sample.open("sample.txt", std::ofstream::app);
		save_sample.open(std::string("./Results/") + name + std::string("_sample.txt"), std::ofstream::app);
		outf_sample << worst << std::endl;
		save_sample << worst << std::endl;

		// Info
		std::ofstream outf_info;
		std::ofstream save_info;
		outf_info.open("info.txt", std::ofstream::app);
		save_info.open(std::string("./Results/") + name + std::string("_info.txt"), std::ofstream::app);
		outf_info << nparticles << " " << iteration << "," << logX_threshold << "," << logL[worst_index] << std::endl;
		save_info << nparticles << " " << iteration << "," << logX_threshold << "," << logL[worst_index] << std::endl;
	}

	// Set the new log-likelihood threshold
	logL_threshold = logL[worst_index];
	std::cout << "Iteration " << iteration << ", Log(X) = " << logX_threshold << " , Log(L) = " << logL_threshold << std::endl;

	// Clone a surviving particle
	if(nparticles != 1)
	{
		// Randomly clone another particle
		int clone = rng.rand_int(nparticles);

		// Make sure we didn't clone the worst particle
		while(clone == worst_index)
		{
			clone = rng.rand_int(nparticles);
		}

		// Replace the worst particle with the worst particle
		particles[worst_index] = particles[clone];
		logL[worst_index] = logL[clone];
		tiebreakers[worst_index] = tiebreakers[clone];
	}

	// Evolve the particle for mcmc_steps so it is not identical to the particle it was cloned from
	int accepted = 0;
	for(int i = 0; i < mcmc_steps; i++)
	{
		// Propose a particle and perturb it
		Particle proposal = particles[worst_index];
		double log_h = proposal.perturb(rng);

		// Makes sure log_h isn't a silly value
		if(log_h > 0.0)
			log_h = 0.0;

		// Log-likelihood of proposed step
		double proposal_logL = proposal.log_likelihood(data);
		//std::cout << "LIKELIHOOD TEST: Particle = " << proposal << ", Likelihood = " << proposal_logL << std::endl;		
		double proposal_tb = fmod(tiebreakers[worst_index] + rng.randh(), 1.0);

		// Compare the proposal with the current worst threshold
		std::vector<double> current { proposal_logL, proposal_tb };
		std::vector<double> threshold { logL_threshold, tb_threshold };

		// Is the evolved particle's log-likelihood above the threshold?
		if((rng.rand() <= exp(log_h)) && is_less_than(threshold, current))
		{
			particles[worst_index] = proposal;
			logL[worst_index] = proposal_logL; 
			tiebreakers[worst_index] = proposal_tb;
			accepted += 1;
		}
	}

	// MCMC diagnostics
	std::cout << "Accepted " << accepted << " of " << mcmc_steps << " MCMC steps." << std::endl;
	//std::cout << "MCMC Particle = " << particles[worst_index] << ", Log-likelihood = " << logL[worst_index] << std::endl;
	//std::cout << std::endl;

	// Output to return
	std::vector<double> output { logX_threshold, logL_threshold };
	//std::cout << output[0] << ", " << output[1] << std::endl;

	return output;
}


// Find the index of the worst particle
template <class T>
int Sampler<T>::find_worst_particle()
{
	// Initialise the worst particle as the first
	int worst_index = 0;
	std::vector<double> worst{ logL[0], tiebreakers[0] };

	// Find the worst particle
	for(int i = 1; i < nparticles; i++)
	{
		// The current particle
		std::vector<double> current { logL[i], tiebreakers[i] };
		
		// Compare current and worst particle
		if(is_less_than(current, worst))
		{
			worst = current;
			worst_index = i;
		}
	}

	// Return the index of the worst particle
	return worst_index;
}



// Sampler - overload <<
// template <class T> - hmm this doesn't work for some reason, have to explcitly define << for Sampler<particle>
std::ostream& operator<<(std::ostream& out, const Sampler<Particle>& sampler)
{
	out << sampler.particles[0];
	return out;
}

// Sampler - print function (mostly so I can see what I've done is working)
template <class T>
void Sampler<T>::print()
{
	std::cout << "The sampler will use " << nparticles << " particles and " << mcmc_steps << " MCMC steps." << std::endl;
	std::cout << "Iteration #" << iteration << std::endl;
	std::cout << "Log-likelihood threshold: " << logL_threshold << std::endl;
	std::cout << "Log-likelihood X threshold: " << logX_threshold << std::endl;
	std::cout << "Log-likelihood tiebreaker threshold: " << tb_threshold << std::endl;
	std::cout << "Particle #1: " << particles[0] << std::endl;
	std::cout << "Log-likelihood #1: " << logL[0] << std::endl;
}



// Compare likelihoods of particles
// Pass two vectors of (likelihood, tiebreaker) as arguments for the current and worst particle
bool is_less_than(std::vector<double> current, std::vector<double> worst)
{
	// Current particle is worse than worst particle
	if(current[0] < worst[0])
		return true;
	// If likelihoods are the same, use tiebreakers
	if(current[0] == worst[0] && current[1] < worst[1])
		return true;
	
	// Otherwise worst particle is still the worst
	return false;
}


// Calculate the log evidence, information, and posterior weights from the output of a run
std::vector<std::vector<double> > calculate_logZ(const std::vector<double> logX, const std::vector<double> logL)
{
	// Storage vectors
	std::vector<double> log_prior(logX.size()); // prior weights
	std::vector<double> log_post(logX.size()); // unnormalised posterior weights
	double logZ; // log-evidence
	std::vector<double> post(logX.size()); // posterior weights
	std::vector<double> H_vec(logX.size()); // vector of information
	double H = 0.0; // information

	// Do logsumexp on the vector of logX
	double logX_store = logsumexp(logX);

	// Calculate for each iteration of the nested sampling run
	for(int i = 0; i < logX.size(); i++)
	{
		log_prior[i] = logX[i] - logX_store;
		log_post[i] = log_prior[i] + logL[i];
	}
	
	// Do logsumexp on the vector of log_post
	logZ = logsumexp(log_post);

	// Calcualte posterior weights
	for(int i = 0; i < logX.size(); i++)
	{
		post[i] = exp(log_post[i] - logZ);
		H_vec[i] = post[i] * (log_post[i] - logZ - log_prior[i]);
		H += H_vec[i]; // sum H_vec to get the information
	}

	// Return logZ, H and log_post
	std::vector<std::vector <double> > store(log_post.size(), std::vector<double> (3));

	// Populate store
	for(int i = 0; i < log_post.size(); i++)
	{
		store[i][0] = logZ;
		store[i][1] = H;
		store[i][2] = log_post[i];
	}
	
	return store;
}


// Do nested sampling function
void do_nested_sampling(const std::vector<std::vector<int> >& data, int nparticles, int mcmc_steps, double max_depth, std::string name, RNG& rng)
{
	// Number of nested sampling iterations
	int steps = nparticles * max_depth;
	
	// Create the Sampler and initialise it
	Sampler<Particle> sampler(data, nparticles, mcmc_steps, name);
	sampler.initialise(rng);
	
	// Storage of results
	std::vector<double> logX(steps);
	std::vector<double> logL(steps);

	// Do the nested sampling
	for(int i = 0; i < steps; i++)
	{
		std::vector<double> ns_run = sampler.do_iteration(rng);

		// Save log-X and log-likelihood thresholds
		logX[i] = ns_run[0];
		logL[i] = ns_run[1];
	}

	// Results and posterior weights
	std::vector<std::vector<double> > results = calculate_logZ(logX, logL);
	std::vector<double> post_weights(logX.size());

	// File to save the posterior weights
	std::ofstream outf("posterior_weights.txt");
	std::ofstream save_weights(std::string("./Results/") + name + std::string("_weights.txt"));

	// Save posterior weights to file
	for(int i = 0; i < results.size(); i++)
	{
		// Calculate weights
		post_weights[i] = exp(results[i][2] - results[i][0]);

		// Save weights to file
		std::ofstream save_weights;		
		save_weights.open(std::string("./Results/") + name + std::string("_weights.txt"), std::ofstream::app);
		
		outf << post_weights[i] << std::endl;
		save_weights << post_weights[i] << std::endl;
	}
}