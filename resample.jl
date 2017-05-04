## Packages
using PyPlot
using StatsBase

## A function that samples from the posterior sample
function resample(player = "", parameter = 1, nsamples = 1000000)
	
	## Filenames
	sampfile = string(player, "sample.txt")
	if(player != "")
		weightfile = string(player, "weights.txt")
	else
		weightfile = string("posterior_weights.txt")
	end
	
	## Posterior weights
	weights = readdlm(weightfile)[:,1] # so it is 1-dimensional
	particles = readdlm(sampfile) # the parameters

	## Sample rows of particles with respective weights
	samp = sample(1:length(particles[:, 1]), WeightVec(weights), nsamples, replace = true)

	## An empty array
	params = Array{Float64}(length(samp))

	## Get the rows of particles in samp
	for(i in 1:length(samp))
		params[i] = particles[samp[i], parameter]
	end
	
	println(string("Median = ", median(params)))
	println(string("Mean = ", mean(params)))
	println(string("Standard Deviation = ", std(params)))
	println(string("Quantiles (95/68/50/68/95)", quantile(params, [0.05, 0.16, 0.5, 0.84, 0.95])))

	## Trying to overplot the prior distributions
	#del = collect(0:0.1:50)
	#d = LogNormal(3.4, 0.7)
	
	## Using PyPlot
	## Re-define the plot parameters
	PyPlot.subplot(1, 1, 1)
	PyPlot.plt[:hist](params, 40)
	#PyPlot.plt.plot(del, pdf(d, del))
	PyPlot.title("Re-weighted posterior sample for mu")
	PyPlot.xlabel("mu")
	PyPlot.ylabel("frequency")
end

## Plot the sample using PyCall to match Brendon's plotting
#plt.ion()
#plt.hold(false)
#plt.hist(params, 50)
#plt.title(string("Plot of posterior values for h"))
#plt.xlabel("h")
#plt.ylabel("frequency")
#plt.ioff()
#plt.show()

## Plot the sample using Plotly
#data = [
#	[
#		"x" => params,
#		"type" => "histogram"
#	]
#]

#response = Plotly.plot(data, ["filename" => "reweighted-posterior", "fileopt" => "overwrite"])
#plot_url = response["url"]