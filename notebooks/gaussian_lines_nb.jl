### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ c13a86de-cb38-11ee-3890-c93e2ad0f39a
# ╠═╡ show_logs = false
begin
	import Pkg
	Pkg.activate("..")
	using LaTeXStrings
	using LinearAlgebra, Plots
	using Revise
end;

# ╔═╡ 9e13dfd5-078d-49bb-827e-97575a6a42df
	push!(LOAD_PATH,"../src/");

# ╔═╡ bb6f403c-0897-4903-be58-8cd320f83d17
begin 
	using blasso, sfw, certificate, toolbox
end

# ╔═╡ 5dcf6210-5e2d-4c74-854e-5617749d8b8c
md"""
# Gaussian Lines Kernel
This Notebook presents the super-resolution of blurred lines from a Gaussian kernel using three examples:
- one with three well separated lines with high noise,
- one with two lines with the same offset and higly close in angle and
- one with seven lines with moderate amount of noise.

Each example shows the computed lines on the original image and in the parameter space. The main algorithm used in all these examples is defined in the function `main_alg` at the bottom of this notebook.
"""

# ╔═╡ 775462f8-59ae-4745-970f-7f5d24bf3dff
md"## Well-Separated Lines"

# ╔═╡ c0ed2392-da39-4ce4-a8b9-58dec586a9d9
begin
	a0=[1., 1., 1.];
	x0=[[0., -π/5], [-15., pi/16], [10., pi/6]];
	lambda=5.;
	sigma_noise = 0.31;
end

# ╔═╡ 432fb5f7-048e-4a31-bad6-c033bfe1a435
x0

# ╔═╡ c6b23a87-166e-4fc7-8cd9-a80b26912753
md"""
## Two Close Lines
"""

# ╔═╡ 95b38abf-a574-4696-a8e8-4b71cc23a4da
begin
	# Initial measure
	a02=[1., 1.];
	x02=[[-1, -0.73], [1, -0.75]];
	sigma_noise2=0.031; # equivalent for max 255 to randn()*noiselevel with noiselevel=20
	lambda2=1.;
end

# ╔═╡ 41f526a1-ee53-49f6-b308-9f051fc1a255
md"""
## Many Intermingled Lines
"""

# ╔═╡ 76cb3f79-5a4a-4397-a254-2c676855ac26
begin
	# Initial measure
	a03=[60, 80, 255, 100, 180, 120, 240]/255;
	x03=[[15, -0.75], [25, -0.5], [2, -0.25], [7, 0.001], [-20, 0.3], [-5, 0.55], [-10, 0.75]];
	sigma_noise3=0.031; # equivalent for max 255 to randn()*noiselevel with noiselevel=20
	lambda3=1.;
end

# ╔═╡ 046e3240-761f-4a89-9fac-aae75357aab0
md"# Misc."

# ╔═╡ 200a7dfa-7270-4c36-ab96-c5ec0c056522
begin
@doc raw"""
`main_alg(x0, a0, sigma_noise, lambda, [kernel_sigma, 
M, 
angle_min,
angle_max]`

Convenience method to run the algorithm.

# Arguments
- `x0, a0` : 
	original image spikes and amplitudes respectively. 
	`x0` is of the form [[offset, angle], ...]
- `sigma_noise` :
	multiplicative constant to the noise
- `lambda`:
	Regulating constant for the BLASSO problem
- `kernel_sigma`: `[1.,1.]` by default
	parameter fixing the height and width of the Gaussian kernel 
- `M` : 32 by default
	half size of the image
- `angle_min, angle_max` : $-\pi/3$ and $\pi/3$ resp. by default
	parameter space bounds 
- `radon` : `true` by default
	Whether to use Radon transform for the first estimate (default) or a minimization on the parameter grid.

# Return
- results : 
	blasso result struct
- comparison : 
	string comparing results with ground truth
- Plots obj : 
	Original Image
- Plots obj : 
	Recovered Lines
- Plots obj : 
	Parameter space spikes
"""
function main_alg(x0::Array{Array{Float64,1},1}, a0::Array{Float64}, sigma_noise::Float64, lambda::Float64; kernel_sigma::Array{Float64}=[1.,1.], M::Float64=32., angle_min::Float64=-π/3, angle_max::Float64=π/3, radon::Bool=true)
	#========================== Main Algorithm ================================#
	# Specifying the Kernel 
	px=range(-M, M);
	px=collect(px);
	py=px;
	N = length(px)*length(py)
	kernel=blasso.setGaussLinesKernel(px,py,kernel_sigma,angle_min,angle_max);
	
	# Noise
	w0=randn(N);
	# Load operator Phi
	operator=blasso.setGaussLineOperator(kernel,a0,x0,sigma_noise*w0, radon);
	
	# Load objective function
	fobj=blasso.setfobj(operator,lambda);
	
	
	# sfw4blasso options
	options=sfw.sfw_options();
	# Computing the results
	result=sfw.sfw4blasso(fobj,kernel,operator,options); # Solve problem

	#=============================== Plotting =================================#
	function plotResult()
		rec_amps, rec_diracs = blasso.decompAmpPos(result.u, d=2)
		max_amp = maximum(rec_amps)
		x = kernel.px
		y = kernel.py
		Npx = kernel.Npx
		Npy = kernel.Npy
	
		############ Plotting the original image #############################
		# Making it
		image_in = zeros(Npx, Npy)
		for i in 1:length(a0)
			image_in += a0[i]*reshape(operator.phi(x0[i]), (Npx,Npy))
		end
		image_in += sigma_noise*reshape(w0, (Npx,Npy))
		# Plotting it
		p_imageIn = heatmap(image_in,
							c=:grays,
							ratio=1,
							cbar=:none,
							framestyle=:none,
							sizes=(1000,1000),
							);
	
		
		############ Original Image with Result Lines on Top #################
		p_lines = plot(sizes=(1000,1000));
		## Plotting the original image
		heatmap!(x,y,image_in, c=:grays, ratio=1);
	
		## Line Plotting
		# Plotting the ground truth 
		for i in 1:length(a0)
			local yt = - tan( π/2 - x0[i][2] ) * ( x .- x0[i][1]);
			plot!(x, yt, 
				  lw=5,
				  c=:black,
				  label="Ground Truth")
		end
		
		# Plotting the recovered lines
		for i in 1:length(rec_amps)
			local y = - tan( π/2 - rec_diracs[i][2] ) * ( x .- rec_diracs[i][1]);
	
			local color = RGBA(1.,0.,0.,
				max(rec_amps[i]/max_amp,.6))
			plot!(x, y, 
				  lw=1.5,
				  c=color,
				  label="Recovered")
		end
		## Misc. options
		plot!(ylim=[y[1], y[end]],
			cbar=:none, 
			framestyle=:none,
			legend=:none
		);
		
		
		############## Parameter Space Plot ##############################
		p_parameterSpace = plot(sizes=(1000,1000));
		# Making the array{array} into  2d arrays
		x0_s = stack(x0)
		rec_dirac_s = stack(rec_diracs)
		# Ground Truth Spikes
		scatter!(x0_s[1,:], x0_s[2,:], 
			markersize=15, 
			c=:black, 
			label="Ground Truth")	
		# Result Spikes
		scatter!(rec_dirac_s[1,:], rec_dirac_s[2,:],
			markersize=6,
			# Intensity of amps are shown by alpha of the marker
			c=RGBA.(1.,0.,0.,max.(rec_amps./max_amp,0.5)),
			label="Recovered")
		
		# Misc. Options
		plot!(
			title=" ",
			xlabel=L"$\eta$",
			ylabel=L"$\theta$",
			ylimit=[-pi/2, pi/2],
			#yticks=((-4:4)*pi/8,[L"-\frac{\pi}{2}", L"-\frac{3\pi}{8}", L"-\frac{\pi}{4}", L"-\frac{\pi}{8}", L"0", L"\frac{\pi}{8}",L"\frac{\pi}{4}",L"\frac{3\pi}{8}",L"\frac{\pi}{2}"]),
			# Setting ticks to multiples of π
			yticks=((-2:2)*pi/4,[L"-\frac{\pi}{2}", L"-\frac{\pi}{4}", L"0",L"\frac{\pi}{4}",L"\frac{\pi}{2}"]),
			labelfontsize=18,
			xtickfontsize=14,
			ytickfontsize=18,
			yguidefontrotation=.9,
			legendfontsize=12,
			minorgrid=true,
			minorticks=2,
			margin_top=(20,:px),
			# Draws a box around the plot
			framestyle=:box)
	
		return p_imageIn, p_lines, p_parameterSpace
	end
	p_imageIn, p_lines, p_parameterSpace = plotResult()

	#======================= Comparison =================================#
	
	a,x = blasso.decompAmpPos(result.u, d=2);
	blasso.computeErrors(x0, a0, x, a, operator)
	
	return result, p_imageIn, p_lines, p_parameterSpace
end
end

# ╔═╡ b902905b-ced7-4b05-98fa-9174c6453d1d
result_3lines, p_3lines, p_3lines_lines, p_3lines_parameterSpace = main_alg(x0, a0, sigma_noise, lambda, radon=true)

# ╔═╡ 8caee775-cf0f-42a4-8ce0-9c66580f5104
plot(p_3lines)

# ╔═╡ 36d771c8-2fcb-42f0-a206-11556222d399
plot(p_3lines_lines)

# ╔═╡ 1a7cd06b-dde1-4f2c-9100-10a18f110402
plot(p_3lines_parameterSpace)

# ╔═╡ 6752ac53-174b-4032-99e0-93686130ba58
result_closelines, p_closelines, p_closelines_lines, p_closelines_parameterSpace = main_alg(x02, a02, sigma_noise2, lambda2, radon=false);

# ╔═╡ 6141e714-46df-4f2a-ad8e-969374f7d7a6
plot(p_closelines)

# ╔═╡ 23a093f0-50f4-430a-abed-65119270e541
plot(p_closelines_lines)

# ╔═╡ 6cb65afe-f7d4-48df-975b-2b25a5faac33
plot(p_closelines_parameterSpace)

# ╔═╡ baab2693-f138-401b-96e2-93485d7079de
result_interLines, p_interLines, p_interLines_lines, p_interLines_parameterSpace = main_alg(x03, a03, sigma_noise3, lambda3, radon=false)

# ╔═╡ a3e79a91-fd7d-4ac0-8519-4cc0e634a276
plot(p_interLines)

# ╔═╡ 89afcd86-f919-47a8-b36d-d464c4f1ecaf
plot(p_interLines_lines)

# ╔═╡ ec4ec66b-7354-42e0-841f-de947dfd0f31
plot(p_interLines_parameterSpace)

# ╔═╡ Cell order:
# ╟─5dcf6210-5e2d-4c74-854e-5617749d8b8c
# ╟─775462f8-59ae-4745-970f-7f5d24bf3dff
# ╠═c0ed2392-da39-4ce4-a8b9-58dec586a9d9
# ╠═8caee775-cf0f-42a4-8ce0-9c66580f5104
# ╠═432fb5f7-048e-4a31-bad6-c033bfe1a435
# ╠═b902905b-ced7-4b05-98fa-9174c6453d1d
# ╠═36d771c8-2fcb-42f0-a206-11556222d399
# ╠═1a7cd06b-dde1-4f2c-9100-10a18f110402
# ╟─c6b23a87-166e-4fc7-8cd9-a80b26912753
# ╠═95b38abf-a574-4696-a8e8-4b71cc23a4da
# ╠═6141e714-46df-4f2a-ad8e-969374f7d7a6
# ╠═6752ac53-174b-4032-99e0-93686130ba58
# ╠═23a093f0-50f4-430a-abed-65119270e541
# ╠═6cb65afe-f7d4-48df-975b-2b25a5faac33
# ╟─41f526a1-ee53-49f6-b308-9f051fc1a255
# ╠═76cb3f79-5a4a-4397-a254-2c676855ac26
# ╠═a3e79a91-fd7d-4ac0-8519-4cc0e634a276
# ╠═baab2693-f138-401b-96e2-93485d7079de
# ╠═89afcd86-f919-47a8-b36d-d464c4f1ecaf
# ╠═ec4ec66b-7354-42e0-841f-de947dfd0f31
# ╟─046e3240-761f-4a89-9fac-aae75357aab0
# ╠═200a7dfa-7270-4c36-ab96-c5ec0c056522
# ╠═c13a86de-cb38-11ee-3890-c93e2ad0f39a
# ╠═bb6f403c-0897-4903-be58-8cd320f83d17
# ╠═9e13dfd5-078d-49bb-827e-97575a6a42df
