function out = simulate_data_varrho(seed, train_fraction)
% SIMULATE_DATA_VARRHO
% Separable spatio-temporal GP: K = Ks .* Kt
% HF: fH(s,t) = rho(s) * fL(s,t) + delta(s,t)
% rho(s) varies by spatial location (station), constant over time.
%
% Output struct fields mirror simulate_data:
%   HF_train, HF_test, LF, HF, train_station_ids, test_station_ids, ...
% plus:
%   rho_loc (Ns x 1) and rho_per_row (N x 1 aligned with HF/LF rows)

  if nargin < 2 || isempty(train_fraction), train_fraction = 0.5; end
  train_fraction = max(0,min(1,train_fraction));
  rng(seed);

  %--------------- grid ----------------
  n_time  = 80;
  n_space = 6;
  Nt = n_time; Ns = n_space^2;
  t  = linspace(0,1,Nt)';

  % stations order (s1,s2)
  svals = (1:n_space)';
  coords_space = [kron(svals, ones(n_space,1)), repmat(svals, n_space, 1)]; % Ns x 2

  % full coords (time-fastest within station)
  s1col = repelem(coords_space(:,1), Nt);
  s2col = repelem(coords_space(:,2), Nt);
  tcol  = repmat(t, Ns, 1);

  coords_tbl = table(s1col, s2col, tcol, 'VariableNames', {'s1','s2','t'});
  coords_tbl.loc_id = repelem((1:Ns)', Nt);

  %--------------- kernels ----------------
  rbf_space = @(X, sigma2, l) sigma2 * exp(-0.5 * sq_dists_scaled(X, l));
  rbf_time  = @(tv, sigma2, l) sigma2 * exp(-0.5 * pdist2(tv(:)/l, tv(:)/l, 'euclidean').^2);

  % target correlations -> length-scales
  delta_t = 1/(Nt-1);
  target_corr_time   = 0.8;
  target_corr_spaceL = 0.72;
  target_corr_spaceD = 0.72;

  to_ell = @(c,d) d / sqrt(-2 * log(max(min(c,0.9999),1e-8)));
  d_space = 1;

  l_space_L = [to_ell(target_corr_spaceL, d_space), to_ell(target_corr_spaceL, d_space)];
  l_space_d = [to_ell(target_corr_spaceD, d_space), to_ell(target_corr_spaceD, d_space)];
  l_time_L  = delta_t / sqrt(-2 * log(target_corr_time));
  l_time_d  = delta_t / sqrt(-2 * log(target_corr_time));

  % amplitudes & noises
  sigma_L2        = 2.0;
  sigma_d2        = 0.8;
  sigma_noise_L   = 0.3;
  sigma_noise_dd2 = 0.7;

  %--------------- build rho(s): varies by station ----------------
  % Normalize spatial coords to [0,1]
  s1 = coords_space(:,1); s2 = coords_space(:,2);
  s1n = (s1 - min(s1)) / max(eps,(max(s1)-min(s1)));
  s2n = (s2 - min(s2)) / max(eps,(max(s2)-min(s2)));

  % Smooth non-constant rho(s) in a controlled range, e.g. [0.2, 1.2]
  % You can tweak these coefficients to make rho more/less variable.
  rho_loc = 0.7 ...
          + 0.35*(s1n - 0.5) ...
          - 0.25*(s2n - 0.5) ...
          + 0.20*sin(2*pi*s1n).*cos(2*pi*s2n);

  % Clip to safe positive range (avoid near-zero or negative)
  rho_loc = max(0.15, min(1.35, rho_loc));

  % expand rho to per-row (time replicated)
  rho_per_row = repelem(rho_loc, Nt);

  %--------------- base covariances ----------------
  K_s_L = rbf_space(coords_space, sigma_L2, l_space_L);  % Ns x Ns
  K_t_L = rbf_time(t,            sigma_L2, l_time_L);    % Nt x Nt
  K_s_d = rbf_space(coords_space, sigma_d2, l_space_d);  % Ns x Ns
  K_t_d = rbf_time(t,             sigma_d2, l_time_d);   % Nt x Nt

  % separable spatio-temporal covariances
  one_t  = ones(Nt);
  one_s  = ones(Ns);

  K_L_full = kron(K_s_L, one_t) .* kron(one_s, K_t_L);
  K_d_full = kron(K_s_d, one_t) .* kron(one_s, K_t_d);

  %--------------- draw processes ----------------
  N = Ns*Nt; jitter = 1e-8;

  dL = mvnrnd(zeros(1,N), K_L_full + jitter*eye(N), 1)';  % LF latent
  eL = sqrt(sigma_noise_L) * randn(N,1);
  fL = dL + eL;

  dd = mvnrnd(zeros(1,N), K_d_full + jitter*eye(N), 1)';  % discrepancy
  eD = sqrt(sigma_noise_dd2) * randn(N,1);
  dd = dd + eD;

  % HF with spatially varying rho(s)
  fH = rho_per_row .* fL + dd;

  %--------------- tables ----------------
  HF_tbl = coords_tbl; HF_tbl.fH = fH;
  LF_tbl = coords_tbl; LF_tbl.fL = fL;

  HF_tbl = sortrows(HF_tbl, {'loc_id','t'});
  LF_tbl = sortrows(LF_tbl, {'loc_id','t'});

  %--------------- split by station ----------------
  rng(seed);
  all_stations = (1:Ns)';
  nStations = Ns;
  nTrain = max(0, min(nStations, floor(train_fraction * nStations)));
  tr_idx = randperm(nStations, nTrain);
  train_stations = all_stations(tr_idx);
  test_stations  = setdiff(all_stations, train_stations);

  is_train_row = ismember(HF_tbl.loc_id, train_stations);
  is_test_row  = ~is_train_row;

  HF_train = HF_tbl(is_train_row, :);
  HF_test  = HF_tbl(is_test_row,  :);

 
  % station lookup table (NOW includes rho_loc)
station_coords = table((1:Ns)', coords_space(:,1), coords_space(:,2), rho_loc, ...
    'VariableNames', {'loc_id','s1','s2','rho_loc'});

%--------------- output ----------------
out = struct( ...
    'HF_train', HF_train, ...
    'HF_test',  HF_test, ...
    'LF',       LF_tbl, ...
    'HF',       HF_tbl, ...
    'K_s_L',    K_s_L, ...
    'K_t_L',    K_t_L, ...
    'train_station_ids', train_stations, ...
    'test_station_ids',  test_stations, ...
    'train_row_idx',     find(is_train_row), ...
    'test_row_idx',      find(is_test_row), ...
    'station_coords',    station_coords, ...
    'rho_loc',           rho_loc, ...
    'rho_per_row',       rho_per_row ...
);

fprintf('Varying rho(s): min=%.3f, max=%.3f, mean=%.3f | train stations=%d/%d\n', ...
    min(rho_loc), max(rho_loc), mean(rho_loc), numel(train_stations), Ns);
end