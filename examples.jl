### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ c13a86de-cb38-11ee-3890-c93e2ad0f39a
# ╠═╡ show_logs = false
begin
	import Pkg
	Pkg.activate(".")
	using LinearAlgebra, Plots
end;

# ╔═╡ 9e13dfd5-078d-49bb-827e-97575a6a42df
	push!(LOAD_PATH,"./src/");

# ╔═╡ bb6f403c-0897-4903-be58-8cd320f83d17
begin
	using Revise
	using blasso, sfw, certificate, toolbox
end

# ╔═╡ 5dcf6210-5e2d-4c74-854e-5617749d8b8c
md"# Gaussian 1D Kernel"

# ╔═╡ 21f334a4-ef50-4e84-82c6-1d75a485d6b5
begin
	# Model constants
	sigmax=.05;K=50;
	# Bounds of domain
	bounds=[-1.0,1.0];
	# Option solver
	options=sfw.sfw_options();
	# Load kernel attributes
	kernel=blasso.setKernel(K,sigmax,bounds);
	println(typeof(kernel))
end

# ╔═╡ b4aea33e-6012-44b5-90ee-960e476382bd
begin
	# Initial measure
	a0=[0.8,0.5,2,0.8];
	x0=[-.5,-.1,.1,.5];
	# Noise
	#srand(1);
	w0=randn(K);
	sigma=.01;
	# Load operator Phi
	op=blasso.setoperator(kernel,a0,x0,sigma*w0);
end

# ╔═╡ 436b02fb-2b8b-4e66-93ca-e344ecd90df0
begin
	lambda=.5;
	# Load objective function
	fobj=blasso.setfobj(op,lambda);
end

# ╔═╡ 01ed0bc2-3c35-4d51-8d31-bb084b592879
# ╠═╡ show_logs = false
result=sfw.sfw4blasso(fobj,kernel,op,options); # Solve problem

# ╔═╡ 3c8fb520-419c-4626-b42c-38c813385179
sfw.show_result(result, options)

# ╔═╡ 9f87f847-e175-4029-8870-eeeba7b6cebd
function plotSpikes(x0,a0,result)
	rec_amps = result.u[1:length(x0)]
	rec_diracs = result.u[length(x0)+1:end]
	plot(rec_diracs, rec_amps, line=:stem, label="Recovered Spikes", ls=:solid, lw=2, color=:red, marker=:circle, markersize=4)
	plot!(x0, a0, line=:stem, label="Original", ls=:dash, color=:black, marker=:circle, markersize=2)
end

# ╔═╡ 46029ffa-e796-4209-a6fb-9ced3d0b2b34
plotSpikes(x0,a0,result)

# ╔═╡ 6e88b632-b8d9-4032-9f47-ed37ac0b25ff
md"# Dirichlet 1D"

# ╔═╡ eabdb719-e027-4de8-b457-ad13ad8e47ce
md"Set the initial measures here !"

# ╔═╡ dd1ba505-190a-432e-8de2-cfc2ed9282ba
begin
# Set the Initial measure
a0_d=[1.0,1.0,.1,.7,.8,.5,.5,.59,1.1,.3];
x0_d=[.2,.3,.35,.5,.7,.1,.13,.62,.77,.9];
# Noise level
end

# ╔═╡ 7f411260-ed6b-482d-8dfc-12814873e6ce
# ╠═╡ show_logs = false
begin
	# Model constants
fc_d=15;K_d=2fc_d+1;
# Option solver
options_dirichlet=sfw.sfw_options(show_mainIter=false,show_newPos=false);
# Load kernel attributes
kernel_dirichlet=blasso.setKernel(fc_d);
w0_d=randn(K_d);
sigma_d=.01;
# Load operator Phi
op_d=blasso.setoperator(kernel_dirichlet,a0_d,x0_d,sigma_d*w0_d);

lambda_d=.3;
# Load objective function
fobj_d=blasso.setfobj(op_d,lambda_d);
# Solving the problem
result_d=sfw.sfw4blasso(fobj_d,kernel_dirichlet,op_d,options_dirichlet);
end

# ╔═╡ 8def417d-ad4f-45ff-8bfd-7c290e4adbe7
sfw.show_result(result_d,options_dirichlet);

# ╔═╡ bb755f36-4676-41ec-bab9-2926a22c8e72
begin
plotSpikes(x0_d, a0_d, result_d)
plot!(legend=(0.5, .98))
end

# ╔═╡ 9c1af8d4-80b9-44dc-819a-b1c92303e2eb
begin
	etaV=certificate.computeEtaV(x0,sign.(a0),op);
	etaL,d1etaL,d2etaL=certificate.computeEtaL(result.u,op,lambda);
end

# ╔═╡ 754e45c0-31a4-4088-8b07-112761a32bc3
let
	tt=0:1:500
	plot(tt, map(etaV,tt))
end

# ╔═╡ Cell order:
# ╟─c13a86de-cb38-11ee-3890-c93e2ad0f39a
# ╟─9e13dfd5-078d-49bb-827e-97575a6a42df
# ╠═bb6f403c-0897-4903-be58-8cd320f83d17
# ╟─5dcf6210-5e2d-4c74-854e-5617749d8b8c
# ╠═21f334a4-ef50-4e84-82c6-1d75a485d6b5
# ╠═b4aea33e-6012-44b5-90ee-960e476382bd
# ╠═436b02fb-2b8b-4e66-93ca-e344ecd90df0
# ╠═01ed0bc2-3c35-4d51-8d31-bb084b592879
# ╠═3c8fb520-419c-4626-b42c-38c813385179
# ╠═9f87f847-e175-4029-8870-eeeba7b6cebd
# ╠═46029ffa-e796-4209-a6fb-9ced3d0b2b34
# ╟─6e88b632-b8d9-4032-9f47-ed37ac0b25ff
# ╟─eabdb719-e027-4de8-b457-ad13ad8e47ce
# ╠═dd1ba505-190a-432e-8de2-cfc2ed9282ba
# ╟─7f411260-ed6b-482d-8dfc-12814873e6ce
# ╠═8def417d-ad4f-45ff-8bfd-7c290e4adbe7
# ╟─bb755f36-4676-41ec-bab9-2926a22c8e72
# ╠═9c1af8d4-80b9-44dc-819a-b1c92303e2eb
# ╠═754e45c0-31a4-4088-8b07-112761a32bc3
