function varargout = crosscorrirr(t1, y1, t2, y2, N, S)
% CROSSCORRIRR irregular timeseries cross-correlation
%
%  Syntax:
%  
%    [xcf, lags, tau, bounds] = crosscorrirr(t1, y1, t2, y2)
%    [xcf, lags, tau, bounds] = crosscorrirr(t1, y1, t2, y2, numLags, numSTD)
%
% Description:
%
%   Compute the sample cross-correlation function (XCF) between irregularly
%   sampled univariate stochastic time series. When called with no output
%   arguments, CROSSCORRIRR plots the XCF sequence with confidence bounds.
%
% Input Arguments:
%
%   t1 - Vector of time points of the first univariate time series.
%
%   y1 - Vector of observations of the first univariate time series
%     corresponding to sample times as given in t1.
%
%   t2 - Vector of time points of the second univariate time series.
%
%   y2 - Vector of observations of the second univariate time series
%     corresponding to sample times as given in t2.
%
% Optional Input Arguments:
%
%   numLags - Positive integer indicating the number of lags of the XCF to
%     compute. If empty or missing, the default is to compute the XCF at
%     lags 0, +/-1, +/-2,...,+/-T, where T is the minimum of 20 or one less
%     than the length of the shortest series.
%
%   numSTD - Positive scalar indicating the number of standard deviations
%     of the sample XCF estimation error to compute assuming y1/y2 are
%     uncorrelated. If empty or missing, the default is numSTD = 2
%     (approximate 95% confidence).
%
% Output Arguments:
%
%   xcf - Sample cross-correlation function between (t1,y1) and (t2,y2).
%     xcf is a vector of length 2*numLags+1 corresponding to lags
%     0, +/-1, +/-2, ..., +/-numLags. The center element of xcf contains
%     the zeroth-lag cross correlation. xcf will be a column vector.
%
%   lags - Vector of lags corresponding to xcf (-numLags to +numLags).
%
%   tau - Resolution of the lags, the i-th lag is at time lag(i) * tau.
%
%   bounds - Two-element vector indicating the approximate upper and lower
%     confidence bounds, assuming the input series are uncorrelated.
%
% Reference:
%
%   [1]  Rehfeld, K., Marwan, N., Heitzig, J. and Kurths, J.  Comparison
%        of correlation analysis techniques for irregularly sampled time
%        series.  Nonlinear Processes in Geophysics, 18/2011, pp. 389-404
%        doi:10.5194/npg-18-389-2011
%
% See also CROSSCORR.

	[rows,columns] = size(t1);
	if ((rows ~= 1) && (columns ~= 1)) || (rows*columns < 2)
		error('sample times (t1) must be a vector.');
	end

	[rows,columns] = size(y1);
	if ((rows ~= 1) && (columns ~= 1)) || (rows*columns < 2)
		error('samples (y1) must be a vector.');
	end

	[rows,columns] = size(t2);
	if ((rows ~= 1) && (columns ~= 1)) || (rows*columns < 2)
		error('sample times (t2) must be a vector.');
	end

	[rows,columns] = size(y2);
	if ((rows ~= 1) && (columns ~= 1)) || (rows*columns < 2)
		error('samples (y2) must be a vector.');
	end

	t1 = t1(:); % Ensure a column vector
	y1 = y1(:); % Ensure a column vector
	t2 = t2(:); % Ensure a column vector
	y2 = y2(:); % Ensure a column vector

	if (length(y1) ~= length(t1))
		error('sample times and samples (t1 and y1) must be of same dimension.');
	end
	if (length(y2) ~= length(t2))
		error('sample times and samples (t2 and y2) must be of same dimension.');
	end

	SS = min(length(y1),length(y2)); % Sample size


	if (nargin < 4)
		error(message('crosscorrirr:UnspecifiedInput'))
	end
	if (nargin > 4 && ~isempty(N))
		if (numel(N) > 1)
			error('Number of lags must be a scalar.');
		elseif ((round(N) ~= N) || (N < 0))
			error('Number of lags must be a non-negative integer.');
		elseif (N > (SS-1))
			error('Number of lags must not exceed the minimum series length minus one.');
		end
	else
		N = min(20, SS - 1);
	end

	if (nargin > 5 && ~isempty(S))
		if (numel(S) > 1)
			error('Number of standard deviations must be a scalar.');
		elseif (S < 0)
			error('Number of standard deviations must be non-negative.');
		end
	else
		S = 2;
	end

	y1 = (y1 - mean(y1)) / std(y1);
	y2 = (y2 - mean(y2)) / std(y2);

	txy = sort([t1; t2]);
	tau = mean(txy(2:end) - txy(1:end-1));
	t1 = t1 ./ tau;
	t2 = t2 ./ tau;

	xcf = zeros(2 * N + 1, 1);
	lags = [-N:N];
	bounds = [S;-S] / sqrt(SS);
	for i = 1:length(lags),
		xcf(i) = x_ccf(lags(i), tau, t1, y1, t2, y2);
	end

	if (nargout == 0)
		lineHandles = stem(lags * tau, xcf, 'filled', 'r-o');
		set(lineHandles(1), 'MarkerSize', 4);
		grid('on');
		xlabel('Lag');
		ylabel('Sample Cross Correlation');
		title('Sample Cross Correlation Function');
		hold('on');

		a = axis;
		plot([a(1) a(1); a(2) a(2)],[bounds([1 1]) bounds([2 2])],'-b');
		plot([a(1) a(2)],[0 0],'-k');
		hold('off');
	else
		varargout = {xcf, lags, tau, bounds};
	end
end

function rho = x_ccf(k, tau, t1, y1, t2, y2)
	nsum = 0;
	dsum = 0;
	rho = 0;
	for i = 1:length(y1),
		j = find(t2 > k + t1(i) - 1.1 & t2 < k + t1(i) + 1.1);
		K = x_krnl_bjoernstad_falck(k - (t2(j) - t1(i)));
		dsum = dsum + sum(K);
		nsum = nsum + sum(y1(i) * (y2(j) .* K));
	end
	rho = nsum / dsum;
end

function v = x_krnl_bjoernstad_falck(ktij)
	d = ktij;
	h = 0.25;
	v = 1/sqrt(2*pi*h) * exp(-d.*d / (2 * h * h));
end
