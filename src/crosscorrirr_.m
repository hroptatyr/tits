function varargout = crosscorrirr(t1, y1, t2, y2, N, S)
% CROSSCORRIRR_ irregular timeseries cross-correlation
%
%  Syntax:
%
%    [xcf, lags, tau, bounds] = crosscorrirr(t1, y1, t2, y2)
%    [xcf, lags, tau, bounds] = crosscorrirr(t1, y1, t2, y2, numLags, numSTD)
%
% Description:
%
%   Compute the sample cross-correlation function (XCF) between irregularly
%   sampled univariate stochastic time series.
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
