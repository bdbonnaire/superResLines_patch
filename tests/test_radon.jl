import Pkg
Pkg.activate("..")
push!(LOAD_PATH,"../src")


using Plots 
using RadonKA
using blasso
using sfw

# %% Creating the Line

# Model constants
sigma=[1., 1.];
M = 32.;
MM = 65;
px=range(-M, M);
px=collect(px);
py=px;
angle_max = π/3;
angle_min = - angle_max;
# Option solver
options=sfw.sfw_options();
# Load kernel attributes
kernel=blasso.setGaussLinesKernel(px,py,sigma,angle_min,angle_max);
println(typeof(kernel))
#
# %% Initial measure
N = length(px)*length(py)
# a0=[60, 80, 255, 100, 180, 120, 240]/255;
# x0=[[15, -0.75], [25, -0.5], [2, -0.25], [7, 0.001], [-20, 0.3], [-5, 0.55], [-10, 0.75]];
a0=[1., 1., 1.];
x0=[[0., -π/5], [-15., pi/16], [10., pi/6]];
# Noise
#srand(1);
w0=randn(N);
sigma_noise=0.031;
# Load operator Phi
op=blasso.setGaussLineOperator(kernel,a0,x0,sigma_noise*w0);
image = zeros((length(px),length(py)))
for i in 1:length(a0)
	image += a0[i]*reshape(op.phi(x0[i]), (length(px),length(py)))
end
image += reshape(sigma_noise*w0, (length(px),length(py)))
pIm = heatmap(image, aspect_ratio=1)
# plot!(x->M*cos(x)+M+1,x->M*sin(x)+M+1,0, 2pi; c=:black, lw=4)
#
# %% Compute the Radon Transform
angles = range(-pi/2, pi/2,200) |> collect

T = sqrt(2)M |> ceil
T = convert(Int64, T)
border_img = zeros((2T,2T))
border_img[(2T-MM)÷2 .+ (1:MM), (2T-MM)÷2 .+ (1:MM)] = image
heatmap(border_img)
radon_t = radon(border_img, angles);
pradon = heatmap(angles, range(-T+1,T-1) , radon_t)
# Computing peak of radon
peak = argmax(vec(radon_t))
peak_v = convert.(Float64, [peak % size(radon_t)[1], (peak ÷ size(radon_t)[1])+1])
# scatter!([peak_v[2]],[peak_v[1]])
plot(pIm, pradon)

# %%

peak_real = copy(peak_v)
peak_real[2] -= 100
peak_real[2] *= -pi/2 * 1/100;
# peak_real[1] = abs(peak_real[1] - T)
peak_real[1] -= T
peak_real[1] /= cos(peak_real[2]);
peak_real
# scatter!([peak_real[2]],[peak_real[1]])
x0

# %%
function find_radon_max(angles::Array{Float64})
	pIm = []
	peak_real = Array{Float64,2}(undef, length(angles), 2)
	i = 1
	for angle in angles
		a0 = [ 255]/255
		x0=[ [30, angle]]
		# Noise
		#srand(1);
		w0=randn(N);
		sigma_noise=0.031;
		# Load operator Phi
		op=blasso.setGaussLineOperator(kernel,a0,x0,sigma_noise*w0);
		# %% Compute the Radon Transform
		radon_angles = range(-pi/2, pi/2,200) |> collect

		image = zeros((length(px),length(py)))
		for i in 1:length(a0)
			image += a0[i]*reshape(op.phi(x0[i]), (length(px),length(py)))
		end
		image += reshape(sigma_noise*w0, (length(px),length(py)))
		push!(pIm, heatmap(image, aspect_ratio=1, cbar=:none))

		T = sqrt(2)M #|> ceil
		TT = convert(Int64, ceil(T))
		border_img = zeros((2TT,2TT))
		border_img[(2TT-MM)÷2 .+ (1:MM), (2TT-MM)÷2 .+ (1:MM)] = image
		radon_t = radon(border_img, radon_angles);
		# Computing peak of radon
		peak = argmax(vec(radon_t))
		peak_v = [peak % size(radon_t)[1], (peak ÷ size(radon_t)[1])+1]
		# scatter!([peak_v[2]],[peak_v[1]])
		peak_real[i,:] = copy(peak_v)
		peak_real[i,2] -= 100
		peak_real[i,2] *= -pi/2 /100;
		# peak_real[i,2] -= sign(peak_real[i,2])*pi/2
		peak_real[i,1] -= T
		println(peak_real[i,1])
		peak_real[i,1] /= cos(peak_real[i,2]);

		i += 1;
	end
	return peak_real, pIm
end

# %%
angles = range(-pi/2, pi/2, length=10) |> collect
radon_points, plots = find_radon_max(angles)
radon_points
titles = Matrix{String}(undef, 1, length(angles))
for i in 1:length(titles)
	titles[i] = @sprintf("%.2f %.2f",radon_points[i,2], angles[i])
end
plot(plots..., layout= (2,5),title=titles)
# %%
#=
Discussion
==========
The Radon Transform supposes a support in the unit disk. 
In our case this meant that lines situated at the corners were not detected with the RT.
To overcome this we added a border of √2M all around the image so that the inner disk of the new image
encapsulates all of the information.
With this the RT recovers all lines at the corners.
=#
