### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ c13a86de-cb38-11ee-3890-c93e2ad0f39a
# ╠═╡ show_logs = false
begin
	import Pkg
	Pkg.activate("..")
	using LinearAlgebra, Plots
	using Random, Distributions
	using LaTeXStrings
end

# ╔═╡ 9e13dfd5-078d-49bb-827e-97575a6a42df
begin
	push!(LOAD_PATH,"../src/");
	push!(LOAD_PATH, "../src/libspec");
end

# ╔═╡ bb6f403c-0897-4903-be58-8cd320f83d17
begin 
	using blasso, sfw, certificate, toolbox
	using Spectrogram: gauss_spectrogram
	using Signals: lin_chirp
end

# ╔═╡ 5dcf6210-5e2d-4c74-854e-5617749d8b8c
md"# Spectrogram Kernel"

# ╔═╡ 7c1515b3-1ffa-4b33-8cb3-737aafe6353f
md"""
This Notebook presents the super-resolution of the spectrogram of linear chirps with four examples:
- one with two well separated lines with no noise,
- the same base signal with added high Gaussian noise,
- An application with two interfering linear chirps with moderate noise and
- two crossing modes with moerate noise.

Each example shows the computed frequency on top of the spectrogram The main algorithm used in all these examples is defined in the function `main_alg` at the bottom of this notebook.
"""

# ╔═╡ 648eff19-26fa-4d90-8968-daaacda95974
md"## Setting up the kernel"

# ╔═╡ d16f9d25-3d42-4625-9e68-3727954740f3
begin
	N = 256;
	sigma = .02;
end

# ╔═╡ 81bdf362-fc97-499a-bb20-addabfe586ce
md"## Two Non-crossing Chirps"

# ╔═╡ 9219e293-79c0-4686-86cf-dd95c12b6bdf
begin
x0_2chirps = [ 200. 0.;
	100. -65.]
	a0_2chirps = [1., 1.]
	lambda_2chirps = 10.;
end

# ╔═╡ 0dec2a4e-dd48-4ad1-9f89-54d63877129b
md"## Same Chirps with High Noise"

# ╔═╡ b132b5b3-b445-4799-866f-26b694d83612
sigma_noise = 1.

# ╔═╡ 2f47da1b-b768-4f51-b83f-e6f67a231e2b
md"## Two Chirps with Interference Patterns"

# ╔═╡ 94711b78-2d7e-4719-852b-637914d06f75
begin
	x0_interf = [ 150. 0.;
				90. 0.]
	a0_interf = [1., 1.]
	lambda_interf = 10.;
	sigma_noise_interf = .2;
end

# ╔═╡ 34a3bc8e-b75e-4797-9db3-db47711f34a0
md"## Two Crossing Chirps"

# ╔═╡ 12ba020b-ef3c-48b0-be13-3f886ec85fd8
begin
	local df = 100.
	x0_crossing = [ (N/2+df) -2df;
				(N/2-df) 2df]
	a0_crossing = [1., 1.]
	lambda_crossing = 10.;
	sigma_noise_crossing = .2;
end

# ╔═╡ 822fce14-20e8-443b-9072-6c38adfbfec3


# ╔═╡ 7d615ebf-902c-4b9f-8ac4-0b40e456f9b1
md"# Etc"

# ╔═╡ 9801aeb9-7067-47a8-acb7-ba9702ecbf0c
md"## Main Algorithm"

# ╔═╡ 4ca2d7f3-c07f-44e3-bde9-5b6ac5c981de

"""
`main_alg(x0, a0, sigma, sigma_noise, lambda, [N, angle_min, angle_max]`

Convenience method to run the algorithm.

# Arguments
- `x0, a0` : original image spikes and amplitudes respectively
	x0 is of the form, [freq c; freq c; ...]
- `sigma` : relative width of the spectrogram's Gaussian window
- `sigma_noise` : standard deviation of Gaussian noise (default is 0 ie no noise)
- `lambda` : blasso parameter
- `N` : Amount of samples for the chirp signal (default is 256)
- `angle_min, angle_max` : parameter space bounds (default is ±π/2*.9)

# Return
- results : blasso result struct
- Plots obj : Original Image
- Plots obj : Recovered Lines
"""
function main_alg(x0::Array{Float64,2}, a0::Array{Float64}, sigma::Float64, lambda::Float64, sigma_noise::Float64=0.; N::Int64=256, angle_min::Float64=-π/2*.9, angle_max::Float64=π/2*.9)
	####################### Main Algorithm #################################
	#================= Creating the spectrogram ===========================#
	
	freqs = x0[:,1]
	offsets = x0[:,2]
	sig = lin_chirp(freqs[1], offsets[1]; N)
	for i in 2:length(freqs)
		sig += lin_chirp(freqs[i], offsets[i]; N)
	end
	
	# Adding white Gaussian noise
	if sigma_noise != 0.
		dist = Normal(0, sigma_noise/2)
		sig += rand(dist, N) + im .* rand(dist, N)
	end

	spec = gauss_spectrogram(sig, sigma) * N^2

	# =================== sfw4blasso ==================================#
	# Setting up the kernel 
	Dpt= 1. /(N-1);
	pt=range(0., 1.,step=Dpt) |> collect;
	N = length(pt)
	pω=0.:N-1 |> collect;
	Dpω=1.;
	kernel = blasso.setSpecKernel(pt,pω,Dpt,Dpω, sigma, angle_min, angle_max);
	# Load operator Phi
	operator = blasso.setSpecOperator(kernel, vec(spec));
	
	fobj = blasso.setfobj(operator,lambda);
	# sfw4blasso
	options = sfw.sfw_options(max_mainIter=2);
	# Computing the results
	result=sfw.sfw4blasso(fobj,kernel,operator,options) 

	###################### Plotting #########################################
	function plotResult_Lines()
		rec_amps, rec_diracs = blasso.decompAmpPos(result.u, d=2)
		max_amp = maximum(rec_amps)
		x = kernel.pt
		y = kernel.pω
		Npx = kernel.Npt
		Npy = kernel.Npω

		p_original = heatmap(spec,
			legend=:none, 
			cbar=:none, 
			framestyle=:none)
		plot!(sizes=(1000,1000))

		
		p_lines = plot(sizes=(1000,1000))

		heatmap!(x,y,spec)
	
		# Line Plotting
		#=
		##Plot the ground truth 
		for i in 1:length(x0)
			local yt = N*tan( x0[i][2] ) *  x[2:end-1] .+ N*x0[i][1];
			plot!(x[1:end-1], yt, lw=5, c=:black, label="Ground Truth")
		end
		=#
		# Plot the recovered lines
		for i in 1:length(rec_amps)
			local y = N*tan( rec_diracs[i][2] ) *  x[2:end-1] .+ N*rec_diracs[i][1];
	
			local color = RGBA(1.,0.,0.,
				max(rec_amps[i]/max_amp,.4))
			plot!(x[2:end-1], y, lw=3, c=color, label="Recovered")
		end
		plot!(ylim=[y[1], y[end]],
			legend=:none, 
			cbar=:none, 
			framestyle=:none, 
			margin=(0,:px),
			)

		return p_original, p_lines
	end
	p_original, p_lines = plotResult_Lines()

	#======================= Comparison =================================#
	
	a,x = blasso.decompAmpPos(result.u, d=2);
	a /= sigma*N^2
	a = sqrt.(a)
	x0_vecvec = [ x0[i,:] for i in 1:length(x0[1,:]) ] ./ N
	blasso.computeErrors(x0_vecvec, a0, x, a, operator)
	
	return result, p_original, p_lines
end

# ╔═╡ b4e9af71-df85-4ca4-933a-93f407b716ce
result_2chirps, p_2chirps_original, p_2chirps_lines = main_alg(x0_2chirps, a0_2chirps, sigma, lambda_2chirps, N=N)

# ╔═╡ d71103a1-8e24-48b5-b6cd-9e9cf7a734a3
plot(p_2chirps_lines)

# ╔═╡ ae78812c-2025-4b73-8c0f-73219416217b
result_noise, p_noise_original, p_noise_lines = main_alg(x0_2chirps, a0_2chirps, sigma, lambda_2chirps, sigma_noise, N=N)

# ╔═╡ 590bce00-5da3-45b5-89e2-c9f509629396
plot(p_noise_lines)

# ╔═╡ 8b4443e6-23c9-4bb4-bca1-b3d6e3858682
result_interf, p_interf_original, p_interf_lines = main_alg(x0_interf, a0_interf, sigma, lambda_interf, sigma_noise_interf, N=N)

# ╔═╡ 3d9e2f7e-d738-44ae-a32f-6526c96c0674
plot(p_interf_lines)

# ╔═╡ 45be427e-20d3-454d-9026-b182ebb088a6
result_crossing, p_crossing_original, p_crossing_lines = main_alg(x0_crossing, a0_crossing, sigma, lambda_interf, sigma_noise_crossing, N=N)

# ╔═╡ ebf866a7-f59c-42b0-b8df-d4773f505c38
plot(p_crossing_lines)

# ╔═╡ afcbb565-ff4d-4070-8666-c2c512cb7e77
md"## Imports"

# ╔═╡ Cell order:
# ╟─5dcf6210-5e2d-4c74-854e-5617749d8b8c
# ╟─7c1515b3-1ffa-4b33-8cb3-737aafe6353f
# ╟─648eff19-26fa-4d90-8968-daaacda95974
# ╠═d16f9d25-3d42-4625-9e68-3727954740f3
# ╠═81bdf362-fc97-499a-bb20-addabfe586ce
# ╠═9219e293-79c0-4686-86cf-dd95c12b6bdf
# ╠═b4e9af71-df85-4ca4-933a-93f407b716ce
# ╠═d71103a1-8e24-48b5-b6cd-9e9cf7a734a3
# ╠═0dec2a4e-dd48-4ad1-9f89-54d63877129b
# ╠═b132b5b3-b445-4799-866f-26b694d83612
# ╠═ae78812c-2025-4b73-8c0f-73219416217b
# ╠═590bce00-5da3-45b5-89e2-c9f509629396
# ╠═2f47da1b-b768-4f51-b83f-e6f67a231e2b
# ╠═94711b78-2d7e-4719-852b-637914d06f75
# ╠═8b4443e6-23c9-4bb4-bca1-b3d6e3858682
# ╠═3d9e2f7e-d738-44ae-a32f-6526c96c0674
# ╠═34a3bc8e-b75e-4797-9db3-db47711f34a0
# ╠═12ba020b-ef3c-48b0-be13-3f886ec85fd8
# ╠═822fce14-20e8-443b-9072-6c38adfbfec3
# ╠═45be427e-20d3-454d-9026-b182ebb088a6
# ╠═ebf866a7-f59c-42b0-b8df-d4773f505c38
# ╟─7d615ebf-902c-4b9f-8ac4-0b40e456f9b1
# ╟─9801aeb9-7067-47a8-acb7-ba9702ecbf0c
# ╠═4ca2d7f3-c07f-44e3-bde9-5b6ac5c981de
# ╟─afcbb565-ff4d-4070-8666-c2c512cb7e77
# ╠═c13a86de-cb38-11ee-3890-c93e2ad0f39a
# ╠═9e13dfd5-078d-49bb-827e-97575a6a42df
# ╠═bb6f403c-0897-4903-be58-8cd320f83d17
