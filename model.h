#ifndef MODEL_H
#define MODEL_H

// Include relevant files
#include "RNG.h"

// Packages
#include <vector>
#include <cmath>

// Define constant Pi
#ifndef M_PI
constexpr double M_PI = 3.141592653589793;
#endif

// Define a Particle class
class Particle
{
private:
	// Members
	std::vector<double> params;

public:
	// Members (public)
	//std::vector<double> params;

	// Constructor
	Particle();

	// Function which generates particles from the prior
	void from_prior(RNG& rng);

	// Function which proposes a jump for one of the particle parameters - returns the likelihood of new parameter compared with old
	double perturb(RNG& rng);

	// Calculate the effective average for a vector of innings
	std::vector<double> effective_average(std::vector<int> scores); // for a vector of scores
	double effective_average(int scores); // for a single case

	// Calculate the hazard function for an innings
	double hazard_function(int scores);

	// Calculate log-likelihood for a particle
	double log_likelihood(const std::vector<std::vector<int> >& data);

	// Function to access the parameters

	// Overload the << operator
	friend std::ostream& operator<< (std::ostream& out, const Particle& particle);

	// A function which prints the Particle (mostly so I can check what I've done works)
	void print();
};


// Rejection sampler to generate parameters from a beta(a, b) distribution
double rejection_sampler(const double a, const double b, RNG& rng);

#endif