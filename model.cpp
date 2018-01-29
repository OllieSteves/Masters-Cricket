// Model definition

// Relevant files
#include "model.h"
#include "RNG.h"

// Packages
#include <iostream>
#include <vector>
#include <math.h> // fmod allows for modulus to be calculated for non-integers
#include <algorithm> // for max_element

// Define the particle class functions
// Particle - default constructor function
// We have 3 parameters (mu1, mu2, L) or (C, mu2, D)
Particle::Particle()
{
	params = std::vector<double>(3);
}

// Particle - from_prior function
void Particle::from_prior(RNG& rng)
{
	// Generate a normal(0, 1)
	double z_norm = rng.randn();
	
	// mu2 ~ Lognormal(25, 0.75) prior
	params[1] = (z_norm * 0.75) + log(25);

	// Beta priors for C and D
	params[0] = rejection_sampler(1.0, 2.0, rng);
	params[2] = rejection_sampler(1.0, 5.0, rng);

	//std::cout << "(" << params[0] << ", " << params[1] << ", " << params[2] << ")" << std::endl; // Print value so we can see whats going on
}

// Particle - perturb function
double Particle::perturb(RNG& rng)
{
	// Define logH = p' - p
	double logH = 0.0;

	// Randomly permute one of the particle parameters
	int i = rng.rand_int(3);

	// Mu1
	if(i == 0)
	{
		// Beta(1, 2) prior
		double a = 1;
		double b = 2;

		// Log-prior before
		logH -= (a - 1) * log(params[i]) + (b - 1) * log(1 - params[i]);

		// Explore parameter space
		params[i] += 1 * rng.randh();
		// Map it onto the [0, 1] interval
		params[i] = fabs(fmod(params[i], 1.0));

		// Log-prior after
		logH += (a - 1) * log(params[i]) + (b - 1) * log(1 - params[i]);
	}

	// Mu2
	if(i == 1)
	{
		// Lognormal prior - technically log(normal)##
		double mu = log(25);
		double sigma = 0.75;

		// Log-prior before
		logH -= -0.5 * log(2 * M_PI) - log(sigma) - (1/(2 * pow(sigma, 2))) * pow(params[i] - mu, 2);
		
		// Explore the parameter space (use a scale of sigma)
		params[i] = fabs(params[i] + sigma * rng.randh());

		// Log-prior after
		logH += -0.5 * log(2 * M_PI) - log(sigma) - (1/(2 * pow(sigma, 2))) * pow(params[i] - mu, 2);
	}

	// L
	if(i == 2)
	{
		// Beta(1, 5) prior
		double a = 1;
		double b = 5;

		// Log-prior before
		logH -= (a - 1) * log(params[i]) + (b - 1) * log(1 - params[i]);

		// Explore parameter space
		params[i] += 1 * rng.randh();
		// Map it onto the [0, 1] interval
		params[i] = fabs(fmod(params[i], 1.0));

		// Log-prior after
		logH += (a - 1) * log(params[i]) + (b - 1)*log(1 - params[i]);
	}

	// Return the log-likelihood of the perturbed parameter
	return logH;
}

// Particle - effective average function (for a vector of scores)
std::vector<double> Particle::effective_average(std::vector<int> scores)
{
	// Take the exponential of  mu_2 it is log-normal (but has been modelled as log(normal))
	double mu2 = exp(params[1]);
	double mu1 = params[0] * mu2;
	double L = params[2] * mu2;
	//double mu1 = params[0];
	//double mu2 = params[1];
	//double L = params[2];

	// Vector to store effective averages for each innings
	std::vector<double> averages(scores.size());

	// Calculate the effective average
	for(int i = 0; i < scores.size(); i++)
	{
		// Column 0 is the runs column
		averages[i] = mu2 + (mu1 - mu2) * exp(-scores[i] / L);
	}
	
	return averages;
}

// Particle - effective average function (for a single score)
double Particle::effective_average(int score)
{
	// Take the exponential of  mu_2 it is log-normal (but has been modelled as log(normal))
	double mu2 = exp(params[1]);
	double mu1 = params[0] * mu2;
	double L = params[2] * mu2;
	//double mu1 = params[0];
	//double mu2 = params[1];
	//double L = params[2];

	// Effective average at specified score score
	double averages = mu2 + (mu1 - mu2) * exp(-score / L);

	return averages;
}

// Particle - hazard function
double Particle::hazard_function(int scores)
{
	return 1 / (effective_average(scores) + 1);
}


// Particle - log_likelihood function
double Particle::log_likelihood(const std::vector<std::vector<int> >& data)
{
	// Test explicit parameter values
	//params[0] = 0.34;
	//params[1] = log(11.15);
	//params[2] = 0.054;

	// Extract scores and out/not outs from the data
	std::vector<int> scores(data.size());
	std::vector<int> outs(data.size());
	for(int i = 0; i < data.size(); i++)
	{
		scores[i] = data[i][0];
		outs[i] = data[i][1];
	}

	// Maximum score
	double max_score = *std::max_element(scores.begin(), scores.end());

	// Vector of scores from 0 to a batsman's maximum score
	std::vector<int> runs(max_score + 1);
	for(int i = 0; i <= max_score; i++)
	{
		runs[i] = i;
	}

	// Likelihood storage vectors
	double cumulative_notout = 0.0; // initialize the cumulative likelihood for not out scores
	//std::vector<double> del(runs.size());
	std::vector<double> log_notout(runs.size(), 0.0); // stores the cumulative likelihoods for surviving until a score
	std::vector<double> log_out(runs.size(), 0.0); // stores the cumulative likelihoods for getting out on a score

	// We want to build a vector which contains log-likelihoods of surviving up until score X (i.e. containing the cumulative hazard function H(x))
	for(int i = 0; i < runs.size(); i++)
	{
		// Get log[1 - H(x)] for each score (for both out and not-out scores) - i.e. the log-probability of surving each score
		// log(H(x)) = -log(mu(x) + 1)
		cumulative_notout += log(1 - hazard_function(runs[i]));
		//del[i] = log(1 - hazard_function(runs[i]));
		log_notout[i] = cumulative_notout;

		// Calculate log(H(x)) for each score - i.e. log-probability of getting out on score X
		//log_out[i] = -log(effective_average(runs[i]) + 1);
		log_out[i] = log(hazard_function(runs[i]));
	}

	// Likelihood calculation
	double lout = 0.0;
	double lnotout = 0.0;

	// Now that we have our log-likelihood vectors, calculate the log-likelihood of the data given the particle
	for(int i = 0; i < scores.size(); i++)
	{
		// Get the log-likelihood of surviving until score X (i.e. log-likelihood of not being dismissed until score X - 1)
		// Only include the log-likelihood of surviving a score of 0 if the batsman scored more than 0 that innings
		if(scores[i] > 0)
		{
			lnotout += log_notout[scores[i] - 1];
		}

		// Add the log-likelihood of getting out on score X if score was out 
		if(outs[i] == 0){
			lout += log_out[scores[i]];
		}
	}

	// Print the likelihood so we can see whats going on
	//std::cout << "L(Out) = " << lout << ", L(Not Out) = " << lnotout << std::endl;
	//std::cout << "Log-Likelihood = " << lnotout + lout << std::endl;

	// Return the log-likelihood
	return lnotout + lout;
}

// Particle - overload << operator
std::ostream& operator<< (std::ostream& out, const Particle& particle)
{
	out << particle.params[0] * exp(particle.params[1]) << "," << exp(particle.params[1]) << "," << particle.params[2] * exp(particle.params[1]);
	return out;
}

// Particle - print function
void Particle::print()
{
	std::cout << "(" << params[0] << ", " << params[1] << ", " << params[2] << ")" << std::endl;
}


// Rejection sampler to generate parameters from a beta(a, b) distribution
double rejection_sampler(const double a, const double b, RNG& rng)
{
    // Sample a point on the x-axis from the proposal distribution [0, 1]
    double x = rng.rand();

    // Draw uniformly from [0, PDF max]
    // In the case of a Beta(a = 1, b) distribution, PDF max = b
    double pdfmax = b;
    
    // Draw from a Uniform[0, PDF max]
    double z_unif = rng.rand();
    double proposal = z_unif * pdfmax;

    // Calculate f(x) - beta distribution
    double fx = tgamma(a + b)/(tgamma(a) * tgamma(b)) * pow(x, a - 1) * pow(1 - x, b - 1);

    // Is f(x) < proposal?
    // Keep
    if(fx < proposal)
    {
        //std::cout << x << " accepted" << std::endl;
        return x;
    }
    else // try again
    {
        //std::cout << x << " rejected, try again." << std::endl;
        rejection_sampler(a, b, rng); // Recursive function!!!
    }
}