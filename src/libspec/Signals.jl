module Signals

export lin_chirp

function lin_chirp(a::Float64,
		c::Float64=0.,
		amplitude::Float64=1.;
		T::Float64=1.,
		N::Int64=512,
		t::Array{Float64}=collect((0:N-1)*T/N),
		return_time::Bool=false)
	lin_chirp = amplitude * exp.(2im*π*(a*t + c/2*t.^2))
	if return_time
		return lin_chirp, t
	else
		return lin_chirp
	end
end

end

