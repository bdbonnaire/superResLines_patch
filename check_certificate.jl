using Pkg
Pkg.activate(".")
push!(LOAD_PATH,"./src/");

using Revise
using Plots, LaTeXStrings
using blasso, sfw, certificate, toolbox

# %%Model constants
σ=.01;
Dpt= 1. /512;
pt=range(0., 1.,step=Dpt) |> collect;
N = length(pt)
pω=0.:N-1 |> collect;
Dpω=1.;
# Bounds of domain
θmin = -π/2 * .9
θmax = π/2 * .9
# Option solver
options=sfw.sfw_options();
# Load kernel attributes
kernel=blasso.setSpecKernel(pt,pω,Dpt,Dpω,σ, θmin, θmax);
kernelT=blasso.setSpecTKernel(pt,pω,Dpt,Dpω,σ, θmin, θmax);
println(typeof(kernel))
#println(kernel.grid)
#
# Initial measure
a0=[1.];
x0T=[[.5, 0]]
x0=[[.5*512, 0]]
# Noise
w0=randn(N^2);
sigma=.000;
# Load operator Phi
op=blasso.setSpecOperator(kernel, a0,x0,sigma*w0);
opT=blasso.setSpecTOperator(kernelT, a0,x0T,sigma*w0);
#
# Compute the image
image = zeros(N,N)
for i in 1:length(a0)
	image += reshape(a0[i] * op.phi(x0[i]), (N,N))
end
imageT = zeros(N,N)
for i in 1:length(a0)
	imageT += reshape(a0[i] * opT.phi(x0T[i]), (N,N))
end
pim = heatmap(image, ratio=:equal, title="Original")
pimT = heatmap(imageT, ratio=:equal, title="Transform")
plot(pim, pimT)
#
# %%Compute ηV
#
etaV = certificate.computeEtaV(x0, sign.(a0), op)
etaV_onGrid = etaV.(kernel.meshgrid)
etaVT = certificate.computeEtaV(x0T, sign.(a0), opT)
etaVT_onGrid = etaV.(kernel.meshgrid)

petaV = heatmap(kernel.grid[1], kernel.grid[2], etaV_onGrid)
petaVT = heatmap(kernel.grid[1], kernel.grid[2], etaV_onGrid)
plot(petaV, petaVT)
plot!(xlabel = L"$\eta$", ylabel=L"$\theta$")

# %%Load objective function
lambda=0.001;
fobj=blasso.setfobj(op,lambda);
println(typeof(fobj))

result=sfw.sfw4blasso(fobj,kernel,op,options); # Solve problem
sfw.show_result(result, op)
