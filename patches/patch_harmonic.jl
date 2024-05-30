"""
This file tests the recovery of first order estimate of IF based on patches
on a quadratic chirp.
"""
import Pkg
Pkg.activate("..")
using LinearAlgebra, Plots
using Random, Distributions
using LaTeXStrings

push!(LOAD_PATH,"../src/");
push!(LOAD_PATH, "../src/libspec");

using blasso, sfw, certificate, toolbox
using Spectrogram: gauss_spectrogram
using Signals: lin_chirp

# %% Creating the spec
N = 256;
sigma = .02;
sigma_noise = 0.8;

center_point = [0.6, 100];
c = N-150;
a = (center_point[2] - c*center_point[1]^2)/(1-2*center_point[1]^2);
b = -2*a*center_point[1];
t = (0:N-1)./N |> collect;
phi = a*t + b/2*t.^2 + c/3*t.^3;
dphi = a .+ b*t .+ c*t.^2;
ddphi = b .+ 2c*t;
sig = exp.(2im*pi*(phi));

# Adding white Gaussian noise
if sigma_noise != 0
	dist = Normal(0, sigma_noise/2);
	sig .+= rand(dist, N) .+ im .* rand(dist, N);
end

spec = gauss_spectrogram(sig, sigma) * N^2;

Dpω = 1.;
Dpt = 1/N;
pω = 1.0:N |> collect;
heatmap(t, pω, spec)
	
# %% Trying it on a window
nb_windows = 4
M = N ÷ nb_windows
lambda = 20.
results = []
for i in 1:nb_windows
	index = (i-1)*M +1:i*M
	println("#################### $index ####################");
	pt_win=t[index]
	kernel = blasso.setSpecKernel(pt_win,pω,Dpt,Dpω, sigma, -pi/2*.9, pi/2*.9);
	# Load operator Phi
	operator = blasso.setSpecOperator(kernel, vec(spec[:,index]));

	fobj = blasso.setfobj(operator,lambda);
	# sfw4blasso
	options = sfw.sfw_options(max_mainIter=1);
	# Computing the results
	push!(results, sfw.sfw4blasso(fobj,kernel,operator,options));
end

# %% Plotting 
p = heatmap(t, pω, spec, size=[800,600])
for j in 1:nb_windows
	index = (j-1)*M +1:j*M;
	rec_amps, rec_diracs = blasso.decompAmpPos(results[j].u, d=2);
	x = t;
	y = pω;
	Npx = N;
	Npy = N;

	# Plot the recovered lines
	y = N*tan( rec_diracs[1][2] ) *  x .+ N*rec_diracs[1][1];

	color=:red
	display(plot!(x[index], y[index], lw=2, c=color, label=false))
end
vline!((1:nb_windows) ./nb_windows, label="window bounds")
plot!([2,3], [-1, -2], label="linear estimation", color=:red)
plot!(xlim=[t[1], t[end]],
	legend=:outerbottom, 
	legendcolumns=2)
