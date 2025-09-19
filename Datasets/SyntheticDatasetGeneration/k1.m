function K = k1(x, y, hyp)

% Extract hyperparameters
sigma = hyp(1);   % Signal variance (in original scale)
theta = hyp(2:end);   % Length-scales for each dimension (in original scale)

% Compute the inverse length-scales directly in the original scale
sqrt_theta = sqrt(theta);

% Squared distance between x and y, scaled by the length-scales
K = sq_dist(diag(1./sqrt_theta) * x', diag(1./sqrt_theta) * y');

% Apply covariance function (no exponential transformation, original scale)
K = sigma * exp(-0.5 * K);



end
