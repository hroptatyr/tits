function [xcf, lag, tau] = crosscorrirr(t1, y1, t2, y2, varargin)
% CROSSCORRIRR irregular timeseries cross-correlation
%
%  Syntax:
%  
%    [xcf, lag, tau] = crosscorrirr(t1, y1, t2, y2)
%    [xcf, lag, tau] = crosscorrirr(t1, y1, t2, y2, numLags)
%
% See also CROSSCORR.

	if (isempty(varargin))
		N = 20;
	else
		N = varargin{1};
	end

	y1 = (y1 - mean(y1)) / std(y1);
	y2 = (y2 - mean(y2)) / std(y2);

	txy = sort([t1, t2]);
	tau = mean(txy(2:end) - txy(1:end-1));
	t1 = t1 / tau;
	t2 = t2 / tau;

	xcf = zeros(2 * N + 1, 1);
	lag = [-N:N];
	for i = 1:length(lag),
		xcf(i) = x_ccf(lag(i), tau, t1, y1, t2, y2);
	end
end

function rho = x_ccf(k, tau, t1, y1, t2, y2)
	nsum = 0;
	dsum = 0;
	rho = 0;
	for i = 1:length(y1),
		for j = find(t2 > k + t1(i) - 1.2 & t2 < k + t1(i) + 1.2),
			K = x_krnl_bjoernstad_falck(k - (t2(j) - t1(i)));
			dsum = dsum + K;
			nsum = nsum + y1(i) * y2(j) * K;
		end
	end
	rho = nsum / dsum;
end

function v = x_krnl_bjoernstad_falck(ktij)
	d = ktij;
	h = 0.25;
	v = 1/sqrt(2*pi*h) * exp(-d.*d / (2 * h * h));
end
