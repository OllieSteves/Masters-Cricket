#ifndef SAMPLER_H
#define SAMPLER_H

// Relevant files
#include "model.h"

// Packages
#include <vector>

// A sampler class Template
template <class T> 
class Sampler
{
private:
	// Data
	std::vector<std::vector<int> > data;

	// Class elements
	int nparticles;
	int mcmc_steps;
	std::vector<T> particles;
	std::vector<double> logL;
	std::vector<double> tiebreakers;

	// Current iteration
	int iteration;

	// Current log-likelihood threshold
	double logX_threshold;
	double logL_threshold;
	double tb_threshold;

	// Name
	std::string name;

public:
	// Default constructor
	Sampler<T> () { };

	// Constructor
	Sampler<T>(std::vector<std::vector<int> > data, int nparticles, int mcmc_steps, std::string name);

	// Inititalise the particles
	void initialise(RNG& rng);

	// Do a nested sampling iteration
	std::vector<double> do_iteration(RNG& rng);

	// Find the index of the worst particle
	int find_worst_particle();

	// Overload the << operator
	friend std::ostream& operator<< (std::ostream& out, const Sampler<T>& sampler);

	// A function which prints the Sampler (mostly so I can check what I've done works)
	void print();
};


// Compare particle likelihoods (current particle and worst particle)
bool is_less_than(std::vector<double> current, std::vector<double> worst);

// Calculate the log evidence, information, and posterior weights from the output of a run
std::vector<std::vector<double> > calculate_logZ(const std::vector<double> logX, const std::vector<double> logL);

// Do a nested sampling run
void do_nested_sampling(const std::vector<std::vector<int> >& data, int nparticles, int mcmc_steps, double max_depth, std::string name, RNG& rng);


#endif