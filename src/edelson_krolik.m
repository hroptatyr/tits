function [xcf, lag, tau] = edelson_krolig(tx, ty, px, py, N)
	px = (px - mean(px)) / std(px);
	py = (py - mean(py)) / std(py);

	txy = sort([tx, ty]);
	tau = mean(txy(2:end) - txy(1:end-1));
	tx = tx / tau;
	ty = ty / tau;

	xcf = zeros(2 * N + 1, 1);
	lag = [-N:N];
	for i = 1:length(lag),
		xcf(i) = x_ccf(lag(i), tau, tx, ty, px, py);
	end
end

function rho = x_ccf(k, tau, tx, ty, px, py)
	nsum = 0;
	dsum = 0;
	rho = 0;
	for i = 1:length(px),
		for j = 1:length(py),
			K = x_krnl_bjoernstad_falck(k, tau, (tx(i) - ty(j)));
			dsum = dsum + K;
			nsum = nsum + px(i) * py(j) * K;
		end
	end
	rho = nsum / dsum;
end

function v = x_krnl_edelson_kronik(k, tau, tij)
	v = abs(k - tij) <= 0.5;
end

function v = x_krnl_bjoernstad_falck(k, tau, tij)
	d = k - tij;
	h = 0.25;
	v = 1/sqrt(2*pi*h) * exp(-d*d / (2 * h * h));
end

function v = x_krnl_green_silverman(k, tau, tij)
	d = k - tij;
	v = 0.5 * exp(-d*d) * sin(-d + tau * pi);
end
