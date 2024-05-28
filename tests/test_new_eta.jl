using Pkg
Pkg.activate(".")
using Plots
# %% Testing the conversion
θv = -π/3
ηv = 1.2

N=512
t = (1:N)/N
yv = ηv .+ t * tan(θv)

θ = atan((yv[end]-ηv)*N) 
η = N*ηv
y = η .+ t * tan(θ)

# %%
visuel = plot(t, yv; xlim=[0,1], ylim=[0,1], framestyle=:box, title="Visuel")
reel = plot(t, y; xlim=[0,1], ylim=[0,N], framestyle=:box, title="Reel")
plot(visuel, reel)
println("thetaV = $θv, θ = $θ")

# %% Testing our φ
function phiVect(x::Array{Float64,1})
	# x is of the form (ηv, θv)
	v=zeros(N^2);
	# index for the loop
	local l=1; 
	local η = x[1]*N
	local θ = atan(tan(x[2]) * N)
	local c = tan(θ)
	local σ = 0.01

	ω = 1:N
	for j in 1:N
	  for i in 1:N
		  ωi=ω[i]
		  tj=t[j]
		  v[l]= σ * (1 + σ ^ 4 * c ^ 2) ^ (-1//2) * exp(-2 * pi * σ ^ 2 * (ωi - η - c * tj) ^ 2 / (1 + σ ^ 4 * c ^ 2))
		l+=1;
	  end
	end
	return v;
end
# %%
phi = reshape(phiVect([ηv, θv]), (N,N))

heatmap(phi)
plot!(0:N-1, y, ylim=[0,N])
