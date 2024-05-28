module STFT
	using FFTW
	using Plots

	export stft, test_stft

		"""
			stft(x, h[, Nbin, method, down_factor, shift])

		Return the short time Fourier transform of a 1D signal `x` with real window `h`.

		Be sure that `length(h)` does not exceed `Nbin`.


		# Arguments
		- `Nbin::Integer=length(x)`: Number of frequency bins.
		- `method::Symbol=:zero`: Method to deal with edges. possible values are `:zero` and `:periodic`.
		- `down_factor::Integer=1`: Only every `down_factor` samples are studied.
		- `shift::Integer=0`: signal is truncated from the start by `shift` samples.

		"""
		function stft(x::Vector{ComplexF64}, h::Vector{Float64}; Nbin::Int64=length(x), method::Symbol=:zero, down_factor::Int64=1, shift::Int64=0)
		Lh = length(h) ÷ 2
		if Nbin  == nothing
			Nbin = length(x)
		end
		if length(h) >= Nbin
			throw(DomainError("lenght(h)=$(length(h))", "The length of the window must not exceed the number of frequency bins Nbin=$Nbin")) 
		end
		if down_factor > Lh 
			throw(DomainError("down_factor=$down_factor", "Downsampling factor is too large : greater than half the window ($Lh)")) 
		end
		if method ∉ [:zero,:periodic]
			throw(DomainError("method=$method", "Provided method is not part of the accepted values [:zero,:periodic]"))
		end

		T = (1+shift):down_factor:length(x) |> collect
		stft = zeros(ComplexF64, (Nbin, length(T)))
		
		if method == :zero
			for t in T
				if length(h) % 2 == 1
					index = ( -min(t,Lh):min(length(x)-t,Lh)-1 ) .+1
				else
					index = (1-min(t,Lh)):min(length(x)-t,Lh)
				end

				stft[1:length(index),t] = x[t.+index] .* h[Lh.+index]
				stft[:,t] = fft(stft[:,t])
			end
		elseif method == :periodic
			if length(h) % 2 == 1
				index = -Lh:Lh
			else
				index = -Lh:Lh-1
			end
			for t in T
				stft[1:length(index),t] = x[rem.((t.+index), length(x), RoundDown).+1] .* h
				stft[:,t] = fft(stft[:,t])
			end
		end
		
		stft = exp.(2im*π .* ((0:Nbin-1)*(T.-1)') / Nbin) .* stft
		stft /= Nbin
		return stft
	end

	function stft(x::Vector{Float64}, h::Array{Float64}; Nbin::Int64=length(x), method::Symbol=:zero, down_factor::Int64=1, shift::Int64=0)
		x = Complex.(x)
		stft(x, h; Nbin, method, down_factor, shift)
	end

	function test_stft(nh=20,n=512)
		t = (1:n)/n
		sig = exp.(2im *pi *300* t)
		h = ones(nh)
		spec = abs.(stft(sig, h, method=:zero)./n ).^2
		heatmap(spec, cbar=:none)
		return spec
	end
end
