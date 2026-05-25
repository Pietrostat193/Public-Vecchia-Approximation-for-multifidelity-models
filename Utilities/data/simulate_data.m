function out = simulate_data(seed, train_fraction)
% SIMULATE_DATA  Draws a separable spatio-temporal GP via K = Ks_full .* Kt_full
% Stations ordered: (1,1),(1,2),...,(n_space,n_space); time varies fastest in each station.
%
% Added outputs:
%   out.test_station_ids  - loc_id of removed (test) stations
%   out.train_station_ids - loc_id of training stations
%   out.test_row_idx      - row indices into out.HF for test rows
%   out.train_row_idx     - row indices into out.HF for train rows
%   out.station_coords    - table mapping loc_id -> (s1,s2)

  if nargin < 2 || isempty(train_fraction), train_fraction = 0.5; end
  train_fraction = max(0,min(1,train_fraction));
  rng(seed);

  %--------------- grid ----------------
    n_time  = 10;
  n_space = 4;
  Nt = n_time; Ns = n_space^2;
  t  = linspace(0,1,Nt)';

  % stations in requested order
  svals = (1:n_space)';                      % integer grid {1..n_space}
  coords_space = [kron(svals, ones(n_space,1)), repmat(svals, n_space, 1)]; % (s1,s2)

  % full coords (time-fastest)
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
  target_corr_time   = 0.8;   % per time step
  target_corr_spaceL = 0.72;  % LF latent NN corr
  target_corr_spaceD = 0.72;  % HF delta  NN corr

  to_ell = @(c,d) d / sqrt(-2 * log(max(min(c,0.9999),1e-8)));
  d_space = 1;                 % neighbor distance on integer grid

  l_space_L = [to_ell(target_corr_spaceL, d_space), to_ell(target_corr_spaceL, d_space)];
  l_space_d = [to_ell(target_corr_spaceD, d_space), to_ell(target_corr_spaceD, d_space)];
  l_time_L  = delta_t / sqrt(-2 * log(target_corr_time));
  l_time_d  = delta_t / sqrt(-2 * log(target_corr_time));

  % Signal amplitudes & noises (SNR)
  sigma_L2        = 2.0;
  sigma_d2        = 0.8;
  sigma_noise_L   = 0.3;
  sigma_noise_dd2 = 0.7;
  rho = 0.6;



  % base covariances
  K_s_L = rbf_space(coords_space, sigma_L2, l_space_L);  % Ns x Ns
  K_t_L = rbf_time(t,            sigma_L2, l_time_L);    % Nt x Nt
  K_s_d = rbf_space(coords_space, sigma_d2, l_space_d);  % Ns x Ns
  K_t_d = rbf_time(t,             sigma_d2, l_time_d);   % Nt x Nt

  % expand to full spatio-temporal covariances (Hadamard/separable form)
  one_t  = ones(Nt);
  one_s  = ones(Ns);
  K_s_full_L  = kron(K_s_L, one_t);
  K_t_full_L  = kron(one_s, K_t_L);
  K_L_full    = K_s_full_L .* K_t_full_L;

  K_s_full_d  = kron(K_s_d, one_t);
  K_t_full_d  = kron(one_s, K_t_d);
  K_d_full    = K_s_full_d .* K_t_full_d;

  %--------------- draw processes ----------------
  N = Ns*Nt; jitter = 1e-8;
  dL = mvnrnd(zeros(1,N), K_L_full + jitter*eye(N), 1)';       % LF latent
  eL = sqrt(sigma_noise_L) * randn(N,1);
  fL = dL + eL;

  dd = mvnrnd(zeros(1,N), K_d_full + jitter*eye(N), 1)';       % HF delta + noise
  eD = sqrt(sigma_noise_dd2) * randn(N,1);
  dd = dd + eD;

  fH = rho * fL + dd;

  %--------------- tables ----------------
  HF_tbl = coords_tbl; HF_tbl.fH = fH;
  LF_tbl = coords_tbl; LF_tbl.fL = fL;
  HF_tbl = sortrows(HF_tbl, {'loc_id','t'});
  LF_tbl = sortrows(LF_tbl, {'loc_id','t'});

  %--------------- split by station ----------------
  rng(seed); % match R-like reseeding
  all_stations = (1:Ns)';                      % identical to unique(HF_tbl.loc_id)
  nStations = Ns;
  nTrain = max(0, min(nStations, floor(train_fraction * nStations)));
  tr_idx = randperm(nStations, nTrain);
  train_stations = all_stations(tr_idx);
  test_stations  = setdiff(all_stations, train_stations);

  is_train_row = ismember(HF_tbl.loc_id, train_stations);
  is_test_row  = ~is_train_row;

  HF_train = HF_tbl(is_train_row, :);
  HF_test  = HF_tbl(is_test_row,  :);

  % station lookup table
  station_coords = table((1:Ns)', coords_space(:,1), coords_space(:,2), ...
      'VariableNames', {'loc_id','s1','s2'});

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
      'station_coords',    station_coords ...
  );

  % quick report
  corr_nn_L = exp(-0.5 * (d_space ./ l_space_L(1)).^2);
  corr_nn_d = exp(-0.5 * (d_space ./ l_space_d(1)).^2);
  fprintf('LF NN corr ≈ %.3f | Δ NN corr ≈ %.3f | rho = %.2f | train stations = %d/%d\n', ...
          corr_nn_L, corr_nn_d, rho, numel(train_stations), Ns);
end

