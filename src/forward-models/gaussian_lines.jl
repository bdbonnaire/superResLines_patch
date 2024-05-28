mutable struct gaussianLines <: discrete
  dim::Int64
  px::Array{Float64,1}
  py::Array{Float64,1}
  p::Array{Array{Float64,1},1}
  Npx::Int64
  Npy::Int64

  nbpointsgrid::Array{Int64,1}
  grid::Array{Array{Float64,1},1}
  meshgrid::Array{Array{Float64, 1}, 2}

  sigma::Array{Float64,1}
  bounds::Array{Array{Float64,1},1}
end

""" setspecKernel(px, py, Dpx, Dpy, σ, angle_min, angle_max)
Sets the kernel structure `spec_lchirp` corresponding to φ: X -> H in the sfw4blasso paper.

In our case, X=[a_min, a_max]x[θ_min, θ_max] where 
- θ_min, θ_max correspond to `angle_min`, `angle_max` and 
- a_min and a_max are computed from θ_min and θ_max.

# Args
- px, py : Array{Float64,1} 
	the time and frequency grid on H 
- Dpx, Dpy : Float64
	The spacing in `px` and `py`
- sigma : Float64
	window width used to compute the spectrogram
- angle_min, angle_max : Float64
	min and max values for the angle. [angle_min, angle_max] must be included in [-π/2, π/2).
"""
function setGaussLinesKernel(px::Array{Float64,1},py::Array{Float64,1}, sigma::Array{Float64,1}, angle_min::Float64, angle_max::Float64)

  dim=2;
  # Sampling of the Kernel
  Npx,Npy=length(px),length(py);
  p=Array{Array{Float64,1}}(undef,Npx*Npy);
  for i in 1:Npy
	  for j in 1:Npx
		  p[(i-1)*Npx+j] = [px[j],py[i]]
	  end
  end

  # Sampling of the parameter space X
  ## Computing the bounds of X
  θ_min = angle_min
  θ_max = angle_max
  a_min = -Npx/2 + Npy/2*tan(θ_min)
  a_max = Npx/2 + Npy/2*tan(θ_max)
  bounds = [[a_min, θ_min], [a_max, θ_max]] # bound of the parameter space, not the image
  println("amin = $a_min et amax = $a_max")
  println("θmin = $θ_min et θmax = $θ_max")
  ## Computing the grid
  nb_points_param_grid = [30, 30]
  println("TEST : Number of points on the grid : $nb_points_param_grid")
  # buiding the grid
  g=Array{Array{Float64,1}}(undef,dim);
  a,b=bounds[1],bounds[2];
  for i in 1:dim
    g[i]=collect(range(a[i], stop=b[i], length=nb_points_param_grid[i]));
  end
  #building the meshgrid
  mg1 = ones(length(g[2])) * g[1]'
  mg2 = g[2] * ones(length(g[1]))'
  meshgrid = vcat.(mg1,mg2) # Q°: plot the meshgrid
  return gaussianLines(dim, px, py, p, Npx, Npy, nb_points_param_grid, g, meshgrid, sigma, bounds)
end

mutable struct operator_gaussLines <: operator
  ker::DataType
  dim::Int64
  sigma::Array{Float64,1}
  bounds::Array{Array{Float64,1},1}

  normObs::Float64

  phi::Function
  d1phi::Function
  d11phi::Function
  d2phi::Function
  y::Array{Float64,1}
  c::Function
  d10c::Function
  d01c::Function
  d11c::Function
  d20c::Function
  d02c::Function
  ob::Function
  d1ob::Function
  d11ob::Function
  d2ob::Function
  correl::Function
  d1correl::Function
  d2correl::Function

  radon::Bool
end

function setGaussLineOperator(
		kernel::gaussianLines,
		a0::Array{Float64,1},
		x0::Array{Array{Float64,1},1},
		w::Array{Float64,1},
		radon::Bool=false)

	"""phiVect(x)
	Given the parameters x=(av, θv), computes the associated spectrogram line.

	av and θv are the "visual" arguments, that is those when we consider our observation
	to be on a window of [0,1]x[0,1]. In reality the frequencies are on [0,kernel.Npy-1]
	"""
  function phiVect(x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy);
	# index for the loop
    local l=1; 
	local η = x[1]
	local θ = x[2]
	local σ = kernel.sigma
	
    for j in 1:kernel.Npx
      for i in 1:kernel.Npy
		  yy=kernel.py[i]
		  xx=kernel.px[j]
      v[l] = sqrt(2) * (pi * (σ[1] ^ 2 * cos(θ) ^ 2 + σ[2] ^ 2 * sin(θ) ^ 2)) ^ (-1//2) * exp(-(sin(θ) * yy + cos(θ) * (xx - η)) ^ 2 / (2 * σ[1] ^ 2 * cos(θ) ^ 2 + 2 * σ[2] ^ 2 * sin(θ) ^ 2)) / 2
        l+=1;
      end
    end
    return v;
  end

  function d1φa(x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy);
	# index for the loop
    local l=1; 
	local η = x[1]
	local θ = x[2]
	local σ = kernel.sigma
	
    for j in 1:kernel.Npx
      for i in 1:kernel.Npy
		yy=kernel.py[i]
		xx=kernel.px[j]
    v[l] = sqrt(2) * cos(θ) * (σ[1] ^ 2 * cos(θ) ^ 2 + σ[2] ^ 2 * sin(θ) ^ 2) ^ (-3//2) * pi ^ (-1//2) * (sin(θ) * yy + cos(θ) * (xx - η)) * exp(-(sin(θ) * yy + cos(θ) * (xx - η)) ^ 2 / (2 * σ[1] ^ 2 * cos(θ) ^ 2 + 2 * σ[2] ^ 2 * sin(θ) ^ 2)) / 2
		l+=1;
      end
    end
    return v;
  end

  function d1φθ(x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy);
	# index for the loop
    local l=1; 
	local η = x[1]
	local θ = x[2]
	local σ = kernel.sigma
	
    for j in 1:kernel.Npx
      for i in 1:kernel.Npy
		yy=kernel.py[i]
		xx=kernel.px[j]
    v[l] = sqrt(2) * (σ[1] ^ 2 * cos(θ) ^ 2 + σ[2] ^ 2 * sin(θ) ^ 2) ^ (-5//2) * (σ[1] ^ 2 * sin(θ) * (σ[1] ^ 2 - 2 * σ[2] ^ 2) * cos(θ) ^ 3 - yy * σ[1] ^ 2 * (xx - η) * cos(θ) ^ 2 + sin(θ) * (-σ[2] ^ 4 * sin(θ) ^ 2 + (σ[1] ^ 2 + (xx - η) ^ 2) * σ[2] ^ 2 - yy ^ 2 * σ[1] ^ 2) * cos(θ) + yy * σ[2] ^ 2 * sin(θ) ^ 2 * (xx - η)) * pi ^ (-1//2) * exp(-(sin(θ) * yy + cos(θ) * (xx - η)) ^ 2 / (2 * σ[1] ^ 2 * cos(θ) ^ 2 + 2 * σ[2] ^ 2 * sin(θ) ^ 2)) / 2
		l+=1;
      end
    end
    return v;
  end


  function d11φ(x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy);
	# index for the loop
    local l=1; 
	local η = x[1]
	local θ = x[2]
	local σ = kernel.sigma
	
    for j in 1:kernel.Npx
      for i in 1:kernel.Npy
		yy=kernel.py[i]
		xx=kernel.px[j]
		v[l] = sqrt(2) * (-(cos(θ) ^ 2 + 1) * sin(θ) ^ 4 * yy * σ[2] ^ 4 - cos(θ) * σ[2] ^ 2 * (cos(θ) ^ 2 * σ[2] ^ 2 - yy ^ 2 + 2 * σ[2] ^ 2) * (xx - η) * sin(θ) ^ 3 + 2 * cos(θ) ^ 2 * yy * (-cos(θ) ^ 2 * σ[1] ^ 2 * σ[2] ^ 2 + (-yy ^ 2 / 2 + σ[2] ^ 2 / 2) * σ[1] ^ 2 + σ[2] ^ 2 * (xx - η) ^ 2) * sin(θ) ^ 2 + (xx - η) * ((σ[1] ^ 4 - 2 * σ[1] ^ 2 * σ[2] ^ 2) * cos(θ) ^ 2 + (-2 * yy ^ 2 - σ[2] ^ 2) * σ[1] ^ 2 + σ[2] ^ 2 * (xx - η) ^ 2) * cos(θ) ^ 3 * sin(θ) - cos(θ) ^ 4 * (σ[1] ^ 2 * cos(θ) ^ 2 - 2 * σ[1] ^ 2 + (xx - η) ^ 2) * yy * σ[1] ^ 2) * exp(-(sin(θ) * yy + cos(θ) * (xx - η)) ^ 2 / (2 * σ[1] ^ 2 * cos(θ) ^ 2 + 2 * σ[2] ^ 2 * sin(θ) ^ 2)) * pi ^ (-1//2) * (σ[1] ^ 2 * cos(θ) ^ 2 + σ[2] ^ 2 * sin(θ) ^ 2) ^ (-7//2) / 2
		l+=1;
      end
    end
    return v;
  end

  function d1phiVect(m::Int64,x::Array{Float64,1})
    if m==1
		return d1φa(x);
    else
		return d1φθ(x);
    end
  end

  function d11phiVect(i::Int64,j::Int64,x::Array{Float64,1})
	  return d11φ(x);
  end

  function d2φa(x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy);
	# index for the loop
    local l=1; 
	local η = x[1]
	local θ = x[2]
	local σ = kernel.sigma
	
    for j in 1:kernel.Npx
      for i in 1:kernel.Npy
		yy=kernel.py[i]
		xx=kernel.px[j]
		v[l] = sqrt(2) * cos(θ) ^ 2 * (σ[1] ^ 2 * cos(θ) ^ 2 + σ[2] ^ 2 * sin(θ) ^ 2) ^ (-5//2) * ((xx - η + σ[1]) * (xx - η - σ[1]) * cos(θ) ^ 2 + 2 * sin(θ) * yy * (xx - η) * cos(θ) + sin(θ) ^ 2 * (yy ^ 2 - σ[2] ^ 2)) * pi ^ (-1//2) * exp(-(sin(θ) * yy + cos(θ) * (xx - η)) ^ 2 / (2 * σ[1] ^ 2 * cos(θ) ^ 2 + 2 * σ[2] ^ 2 * sin(θ) ^ 2)) / 2

		l+=1;
      end
  end
    return v;
  end

  function d2φθ(x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy);
	# index for the loop
    local l=1; 
    local η = x[1]
    local θ = x[2]
	local σ = kernel.sigma
	
	local d1φθtemp = d1φθ(x) # Q°: à quoi ça sert ?
	d1φθtemp /= kernel.Npy / (cos(x[2])^2 + kernel.Npy * sin(x[2])^2)
    for j in 1:kernel.Npx
      for i in 1:kernel.Npy
		yy=kernel.py[i]
		xx=kernel.px[j]
		v[l] = sqrt(2) * pi ^ (-1//2) * exp(-(sin(θ) * yy + cos(θ) * (xx - η)) ^ 2 / (2 * σ[1] ^ 2 * cos(θ) ^ 2 + 2 * σ[2] ^ 2 * sin(θ) ^ 2)) * ((cos(θ) ^ 2 + 1) * sin(θ) ^ 6 * σ[2] ^ 8 - 4 * cos(θ) * yy * σ[2] ^ 6 * (xx - η) * sin(θ) ^ 5 - 4 * ((3//2 * σ[1] ^ 4 - σ[1] ^ 2 * σ[2] ^ 2) * cos(θ) ^ 4 + (-yy ^ 2 * σ[1] ^ 2 + σ[2] ^ 2 * (xx - η) ^ 2) * cos(θ) ^ 2 - (yy + σ[2]) * (yy - σ[2]) * (σ[1] ^ 2 + (xx - η) ^ 2) / 4) * σ[2] ^ 4 * sin(θ) ^ 4 + 2 * (xx - η) * cos(θ) * yy * σ[2] ^ 2 * (-2 * cos(θ) ^ 2 * σ[1] ^ 2 * σ[2] ^ 2 + (-yy ^ 2 + 4 * σ[2] ^ 2) * σ[1] ^ 2 + σ[2] ^ 2 * (xx - η) ^ 2) * sin(θ) ^ 3 + cos(θ) ^ 2 * ((8 * yy ^ 2 * σ[1] ^ 4 * σ[2] ^ 2 - 8 * (xx - η) ^ 2 * σ[2] ^ 4 * σ[1] ^ 2) * cos(θ) ^ 2 + (yy ^ 4 - 4 * yy ^ 2 * σ[2] ^ 2) * σ[1] ^ 4 - 4 * σ[2] ^ 2 * (yy - σ[2]) * (yy + σ[2]) * (xx - η) ^ 2 * σ[1] ^ 2 + σ[2] ^ 4 * (xx - η) ^ 4) * sin(θ) ^ 2 - 2 * (xx - η) * ((2 * σ[1] ^ 4 - 2 * σ[1] ^ 2 * σ[2] ^ 2) * cos(θ) ^ 2 + (-yy ^ 2 - 2 * σ[2] ^ 2) * σ[1] ^ 2 + σ[2] ^ 2 * (xx - η) ^ 2) * cos(θ) ^ 3 * yy * σ[1] ^ 2 * sin(θ) - 4 * cos(θ) ^ 4 * ((σ[1] ^ 4 / 4 - σ[1] ^ 2 * σ[2] ^ 2) * cos(θ) ^ 4 + (-σ[1] ^ 4 / 2 + (-yy ^ 2 + 2 * σ[2] ^ 2) * σ[1] ^ 2 + σ[2] ^ 2 * (xx - η) ^ 2) * cos(θ) ^ 2 + (5//4 * yy ^ 2 - 3//4 * σ[2] ^ 2) * σ[1] ^ 2 - (yy ^ 2 + 5 * σ[2] ^ 2) * (xx - η) ^ 2 / 4) * σ[1] ^ 4) * (σ[1] ^ 2 * cos(θ) ^ 2 + σ[2] ^ 2 * sin(θ) ^ 2) ^ (-9//2) / 2

		l+=1;
      end
    end
    return v;
end

  function d2phiVect(m::Int64,x::Array{Float64,1})
    if m==1
		return d2φa(x);
    else
		return d2φθ(x);
    end
  end

  c(x1::Array{Float64,1},x2::Array{Float64,1})=dot(phiVect(x1),phiVect(x2));

  function d10c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(d1phiVect(i,x1),phiVect(x2));
  end
  function d01c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(phiVect(x1),d1phiVect(i,x2));
  end
  function d11c(i::Int64,j::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    if i==1 && j==2 || i==2 && j==1
      return dot(d1phiVect(1,x1),d1phiVect(1,x2));
    end
    if i==1 && j==3 || i==3 && j==1
      return dot(d11phiVect(1,2,x1),phiVect(x2));
    end
    if i==1 && j==4 || i==4 && j==1
      return dot(d1phiVect(1,x1),d1phiVect(2,x2));
    end

    if i==2 && j==3 || i==3 && j==2
      return dot(d1phiVect(2,x1),d1phiVect(1,x2));
    end
    if i==2 && j==4 || i==4 && j==2
      return dot(phiVect(x1),d11phiVect(1,2,x2));
    end

    if i==3 && j==4 || i==4 && j==3
      return dot(d1phiVect(2,x1),d1phiVect(2,x2));
    end
  end
  function d20c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(d2phiVect(i,x1),phiVect(x2));
  end
  function d02c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(phiVect(x1),d2phiVect(i,x2));
  end


  y=sum([a0[i]*phiVect(x0[i]) for i in 1:length(x0)])+w;
  normObs=.5*norm(y)^2;

  function ob(x::Array{Float64,1},y::Array{Float64,1}=y)
    return dot(phiVect(x),y);
  end
  function d1ob(k::Int64,x::Array{Float64,1},y::Array{Float64,1}=y)
    return dot(d1phiVect(k,x),y);
  end
  function d11ob(k::Int64,l::Int64,x::Array{Float64,1},y::Array{Float64,1}=y)
    return dot(d11phiVect(k,l,x),y);
  end
  function d2ob(k::Int64,x::Array{Float64,1},y::Array{Float64,1}=y)
    return dot(d2phiVect(k,x),y);
  end

  # for i in eachindex(kernel.meshgrid)
  #   println(kernel.meshgrid[i])
  # end

  # TODO: tester si mesh grid fonctionne
  PhisY=zeros(prod(kernel.nbpointsgrid));
  l=1;
  for pg in kernel.meshgrid
	  PhisY[l]=ob(pg);
	  l +=1;
  end

  function correl(x::Array{Float64,1},Phiu::Array{Array{Float64,1},1})
	  return dot(phiVect(x),sum(Phiu)-y);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Array{Float64,1},1})
	  d11 = dot(d1phiVect(1,x),sum(Phiu)-y);
	  d12 = dot(d1phiVect(2,x),sum(Phiu)-y);
	  return [d11,d12];
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Array{Float64,1},1})
    d2c=zeros(kernel.dim,kernel.dim);
	d2c[1,2]=dot(d11phiVect(0,0,x),sum(Phiu)-y);
	d2c=d2c+d2c';
    d2c[1,1]=dot(d2phiVect(1,x),sum(Phiu)-y);
	d2c[2,2]=dot(d2phiVect(2,x),sum(Phiu)-y);
    return(d2c)
  end

  operator_gaussLines(typeof(kernel),kernel.dim,kernel.sigma,kernel.bounds,normObs,phiVect,d1phiVect,d11phiVect,d2phiVect,y,c,d10c,d01c,d11c,d20c,d02c,ob,d1ob,d11ob,d2ob,correl,d1correl,d2correl, radon);
end

function computePhiu(u::Array{Float64,1},op::blasso.operator_gaussLines)
  a,X=blasso.decompAmpPos(u,d=op.dim);
  Phiu = [a[i]*op.phi(X[i]) for i in 1:length(a)];
  # Phiux=[a[i]*op.phix(X[i][1]) for i in 1:length(a)];
  # Phiuy=[op.phiy(X[i][2]) for i in 1:length(a)];
  return Phiu;
end

# Compute the argmin and min of the correl on the grid.
function minCorrelOnGrid(Phiu::Array{Array{Float64,1},1},kernel::blasso.gaussianLines,op::blasso.operator_gaussLines,positivity::Bool=true)
  correl_min,argmin=Inf,zeros(op.dim);
  for pg in kernel.meshgrid
	  buffer = op.correl(pg, Phiu)
      if !positivity
        buffer=-abs(buffer);
      end
      if buffer<correl_min
        correl_min=buffer;
        argmin=pg;
      end
  end

  return argmin,correl_min
end

using RadonKA
"""
Use the Radon transform to estimate the position of the next line.
!!!! Careful, this method only works on square images for now !!!!
"""
function radonLineEstimate(Phiu::Array{Array{Float64,1},1},kernel::blasso.gaussianLines,op::blasso.operator_gaussLines,positivity::Bool=true)

	# half-size of the image
	M = kernel.Npx ÷ 2
	# full size of the image
	MM = 2M + 1
	image = reshape(op.y - sum(Phiu), (MM,MM))
	Tf = sqrt(2)M 
	T = convert(Int64, ceil(Tf))
	# making a border of zeros around phiu so that RadonT gets all info
	border_img = zeros((2T,2T))
	border_img[(2T-MM)÷2 .+ (1:MM), (2T-MM)÷2 .+ (1:MM)] = image

	angles = range(-pi/2, pi/2, 200) |> collect
	radon_t = radon(border_img, angles);
	# Computing peak of radon
	peak = argmax(vec(radon_t))
	peak_real = convert.(Float64, [peak % size(radon_t)[1], (peak ÷ size(radon_t)[1])+1])
	peak_real[2] -= 100
	peak_real[2] *= -pi/200;
	# peak_real[1] = abs(peak_real[1] - T)
	peak_real[1] -= Tf
	peak_real[1] /= cos(peak_real[2]);
	println("TEST The estimation from the Radon transform is $peak_real")
	
	return peak_real, op.correl(peak_real, Phiu)
end

"""
Sets the amplitude bounds 
"""
function setbounds(op::blasso.operator_gaussLines,positivity::Bool=true,ampbounds::Bool=true)
  x_low=op.bounds[1];
  x_up=op.bounds[2];

  if ampbounds
    if positivity
      a_low=0.0;
    else
      a_low=-Inf;
    end
    a_up=Inf;
    return a_low,a_up,x_low,x_up
  else
    return x_low,x_up
  end
end


function plotResult_GaussianLines(x0, a0, w, result, kernel, op)
	rec_amps, rec_diracs = blasso.decompAmpPos(result.u, d=2)
	max_amp = maximum(rec_amps)
	x = kernel.px
	y = kernel.py
	Npx = kernel.Npx
	Npy = kernel.Npy

	p_lines = plot()
	image_in = zeros(Npx, Npy)
	for i in 1:length(a0)
		image_in += a0[i]*reshape(op.phi(x0[i]), (Npx,Npy))
	end
	image_in += reshape(w, (Npx,Npy))
	heatmap!(x,y,image_in, c=:grays, ratio=1)

	## Line Plotting
	# Plot the ground truth 
	for i in 1:length(a0)
		local yt = - tan( π/2 - x0[i][2] ) * ( x .- x0[i][1]);
		plot!(x, yt, lw=5, c=:black, label="Ground Truth")
	end
	# Plot the reocovered lines
	for i in 1:length(rec_amps)
		local y = - tan( π/2 - rec_diracs[i][2] ) * ( x .- rec_diracs[i][1]);

		local color = RGBA(1.,0.,0.,
			max(rec_amps[i]/max_amp,.4))
		plot!(x, y, lw=1.5, c=color, label="Recovered")
	end
	plot!(ylim=[y[1], y[end]],
		legend=:none, 
		cbar=:none, 
		framestyle=:none,
		title="Lines",
		titlefontsize=22)

	## Parameter Space Plot
	p_parameterSpace = plot();
	x0_s = stack(x0)
	rec_dirac_s = stack(rec_diracs)
	scatter!(x0_s[1,:], x0_s[2,:], 
		markersize=15, 
		c=:black, 
		label="Ground Truth")	
	scatter!(rec_dirac_s[1,:], rec_dirac_s[2,:],
		markersize=6,
		c=RGBA.(1.,0.,0.,max.(rec_amps./max_amp,0.5)),
		label="Recovered")
	
	plot!(title="Spikes in the Parameter Space",
		titlefontsize=18,
		xlabel=L"$a$",
		ylabel=L"$\theta$",
		ylimit=[-pi/2, pi/2],
		#yticks=((-4:4)*pi/8,[L"-\frac{\pi}{2}", L"-\frac{3\pi}{8}", L"-\frac{\pi}{4}", L"-\frac{\pi}{8}", L"0", L"\frac{\pi}{8}",L"\frac{\pi}{4}",L"\frac{3\pi}{8}",L"\frac{\pi}{2}"]),
		yticks=((-2:2)*pi/4,[L"-\frac{\pi}{2}", L"-\frac{\pi}{4}", L"0",L"\frac{\pi}{4}",L"\frac{\pi}{2}"]),
		labelfontsize=18,
		xtickfontsize=14,
		ytickfontsize=18,
		yguidefontrotation=.9,
		legendfontsize=12,
		minorgrid=true,
		minorticks=2,
		framestyle=:box)

	plot(p_lines, p_parameterSpace)
	plot!(sizes=(1000,500))
end
