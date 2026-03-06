%% reviewer_decomp_vecchia_experiment_using_ModelInfo.m
% Comparison:
%   1) Exact GP                 -> likelihood2Dsp
%   2) Full-MF Vecchia          -> nlml_vecchia_fullMF
%   3) GLS Vecchia              -> likelihoodVecchia_GLS
%
% Uses ONLY "Corr" conditioning
% Sweeps neighbor size (nn_size)

clear; clc;
rng(12345);

global ModelInfo
ModelInfo = struct();

%% -------------------- 0) Load Data --------------------
capN     = 2500;
dataFile = "C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\Projects\6.VecchiaApproximation\3d_SpaceTimeVecchia\1_South_lombardy\South_Lombardy_sorted_data.mat";

S    = load(dataFile);
data = S.sorted_data;

X_L = [data.Time, data.Lat_LF, data.Lon_LF];
X_H = [data.Time, data.Lat_LF, data.Lon_LF];  % adapt if HF coords exist
y_L = data.Wind_speed;
y_H = data.ws;

ModelInfo.X_L = X_L(1:capN,:);
ModelInfo.X_H = X_H(1:capN,:);
ModelInfo.y_L = y_L(1:capN);
ModelInfo.y_H = y_H(1:capN);

ModelInfo.kernel       = "RBF";
ModelInfo.cov_type     = "RBF";
ModelInfo.jitter       = 1e-5;
ModelInfo.MeanFunction = "zero";
ModelInfo.combination="multiplicative";
ModelInfo.RhoFunction="constant";

ModelInfo.conditioning   = "Corr";   % FIXED
ModelInfo.nn_size        = 20;
ModelInfo.cand_mult      = 50;
ModelInfo.usePermutation = true;
ModelInfo.show_path_diag = false;

fprintf('\n====================================================\n');
fprintf('      EXACT vs FullMF Vecchia vs GLS Vecchia\n');
fprintf('====================================================\n');

%% -------------------- 1) Hyperparameters --------------------
if isfield(ModelInfo,'hyp_current') && ~isempty(ModelInfo.hyp_current)
    hyp_base = ModelInfo.hyp_current;
elseif isfield(ModelInfo,'hyp') && ~isempty(ModelInfo.hyp)
    hyp_base = ModelInfo.hyp;
else
    hyp_base = 0.1*ones(14,1);   % adjust if needed
    fprintf('WARNING: using default hyperparameters\n');
end

fprintf('Using hyp of length %d\n', numel(hyp_base));

%% -------------------- 2) EXACT BASELINE --------------------
fprintf('\n=== EXACT (likelihood2Dsp) ===\n');

NL_full = likelihood2Dsp(hyp_base);

alpha_exact  = ModelInfo.alpha;
logdet_exact = 2 * ModelInfo.log_det_classic;

y_joint = [ModelInfo.y_L; ModelInfo.y_H];
N = numel(y_joint);

fprintf('Exact NLML: %.6f\n', NL_full);

%% -------------------- 3) Sweep nn_size --------------------
sizes = [10 15 20 25 30 40 60];

RES = table();
row = 0;

fprintf('\n=== Vecchia Sweep (Corr conditioning only) ===\n');

for i = 1:numel(sizes)

    nn = sizes(i);
    ModelInfo.nn_size = nn;
    ModelInfo.conditioning = "Corr";

    % Clear Vecchia caches
    fieldsToClear = {'vecchia_idxL','vecchia_idxH','SIy','y_tilde', ...
                     'H','L','A','D_inv','R','perm','hyp_current'};
    for f = 1:numel(fieldsToClear)
        if isfield(ModelInfo, fieldsToClear{f})
            ModelInfo = rmfield(ModelInfo, fieldsToClear{f});
        end
    end

    %% ---- FullMF Vecchia ----
    t1 = tic;
    NL_fullMF = nlml_vecchia_fullMF(hyp_base);
    time_fullMF = toc(t1);

    alpha_fullMF = ModelInfo.SIy;
    y_tilde      = ModelInfo.y_tilde;

    relErr_alpha_fullMF = norm(alpha_fullMF - alpha_exact) / max(norm(alpha_exact),1e-12);

    quad_fullMF   = y_tilde' * alpha_fullMF;
    logdet_fullMF = 2*(NL_fullMF - 0.5*quad_fullMF - 0.5*N*log(2*pi));
    relErr_logdet_fullMF = abs(logdet_fullMF - logdet_exact) / max(abs(logdet_exact),1e-12);

    %% ---- GLS Vecchia ----
    t2 = tic;
    NL_GLS = likelihoodVecchia_nonstat_GLS(hyp_base);
    time_GLS = toc(t2);

    alpha_GLS = ModelInfo.SIy;
    y_tilde_GLS = ModelInfo.y_tilde;

    relErr_alpha_GLS = norm(alpha_GLS - alpha_exact) / max(norm(alpha_exact),1e-12);

    quad_GLS   = y_tilde_GLS' * alpha_GLS;
    logdet_GLS = 2*(NL_GLS - 0.5*quad_GLS - 0.5*N*log(2*pi));
    relErr_logdet_GLS = abs(logdet_GLS - logdet_exact) / max(abs(logdet_exact),1e-12);

    %% ---- Store ----
    row = row + 1;
    RES.m(row,1) = nn;

    RES.NLML_FullMF(row,1) = NL_fullMF;
    RES.NLML_GLS(row,1)    = NL_GLS;

    RES.AbsDiff_FullMF(row,1) = NL_fullMF - NL_full;
    RES.AbsDiff_GLS(row,1)    = NL_GLS - NL_full;

    RES.RelErrAlpha_FullMF(row,1) = relErr_alpha_fullMF;
    RES.RelErrAlpha_GLS(row,1)    = relErr_alpha_GLS;

    RES.RelErrLogdet_FullMF(row,1) = relErr_logdet_fullMF;
    RES.RelErrLogdet_GLS(row,1)    = relErr_logdet_GLS;

    RES.Time_FullMF(row,1) = time_fullMF;
    RES.Time_GLS(row,1)    = time_GLS;

    fprintf('m=%3d | FullMF ΔNL=%.3e | GLS ΔNL=%.3e\n', ...
        nn, NL_fullMF-NL_full, NL_GLS-NL_full);
end

%% -------------------- 4) Display Results --------------------
fprintf('\n=== FINAL RESULTS ===\n');
disp(RES);
