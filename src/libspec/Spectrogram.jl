module Spectrogram
	using STFT
	export spectrogram, gauss_spectrogram

	"""
		spectrogram(f, window; kwargs...)
	
	Compute the spectrogram of `f` with window `window`.

	kwargs are that of `STFT.stft`.
	"""
	function spectrogram(f::Array{ComplexF64}, window::Array{Float64}; kwargs...)
		f_stft = stft(f, window; kwargs...)
		spec = abs.(f_stft) .^ 2
	end

"""
	gauss_spectrogram(f, sigma[, Nbin, step_size, precision, return_window, kwargs...])

Return the spectrogram of `f` with window ``g(x)= e^{-π x²/sigma²}``. 

Discretization of the window follows that of the signal and the points ``g[k]`` kept
are such that ``g[k]>= prec``.

# Arguments
- `Nbin::Integer=length(f)`: number of frequency bins in the FFT.
- `step_size::Float=1/N`: step size of `f`'s discretization.
- `prec::Float=1e-3`: The support of the window is aproximated to `` g(x) >= prec``.
- `return_window::Boolean=false`
- `kwargs`: arguments for STFT.stft.

# Return
- spectrogram
- [window]: Gaussian window used, if `return_window` is set.
"""
	function gauss_spectrogram(f::Array{ComplexF64}, sigma::Float64;
			Nbin::Int64=length(f),
			step_size::Float64=1/length(f),
			precision::Float64=1e-3,
			return_window::Bool=false,
			kwargs...)
		l = floor( sigma / step_size * √(-log(precision) / π))
		T = (-l:l) * step_size
		window = exp.(-π * (T/sigma).^2)
		spec = spectrogram(f, window; Nbin, kwargs...)

		if return_window
			return spec, window
		else
			return spec
		end
	end
end
