
function sim_show_regime_switch()
% SIM_SHOW_REGIME_SWITCH
% Demonstrates when Full-K Vecchia vs CM/Nested Vecchia is more accurate,
% under controlled simulation regimes (sparse HF + increasing rho).
%
% Requires you have on path:
%   simulate_data_dynamic
%   likelihood2Dsp
%   nlml_vecchia_fullMF
%   likelihoodVecchia_nonstat_GLS_v4   (or your CM vecchia)

global ModelInfo

% -------------------- EXPERIMENT SETTINGS --------------------
seeds_data   = 1:5;            % repeat datasets for robustness
n_trials_hyp = 20;             % random hyper draws per dataset-condition
trainF_list  = [0.05 0.10 0.20];
rho_list     = [0.9];
nn_list      = [5 10 20];

% hyper perturbation around center (kept moderate to avoid tons of PD fails)
scale_log = 0.45;
scale_rho = 0.10;

% -------------------- SIMULATION CONFIG --------------------
cfg0 = struct();
cfg0.n_time  = 30;
cfg0.n_space = 10;
cfg0.target_corr_time   = 0.97;
cfg0.target_corr_spaceL = 0.95;
cfg0.target_corr_spaceD = 0.80;
cfg0.sigma_L2        = 2.0;
cfg0.sigma_d2        = 0.5;
cfg0.sigma_noise_L   = 0.05;
cfg0.sigma_noise_dd2 = 0.10;

fprintf('\n====================================================================\n');
fprintf(' REGIME SWITCH: Full-K Vecchia vs CM/Nested Vecchia (vs FULL exact)\n');
fprintf(' Metric: median relative NLML error |Δ|/|NL_full| (lower is better)\n');
fprintf('====================================================================\n');

for trainF = trainF_list

    fprintf('\n======================= train_fraction = %.2f =======================\n', trainF);

    for nn_size = nn_list

        fprintf('\n---- nn_size = %d ----\n', nn_size);
        fprintf('%6s | %14s %6s | %14s %6s\n', ...
            'rho', 'medRel(FULLK)', 'win', 'medRel(CM)', 'win');
        fprintf('%s\n', repmat('-',1,54));

        for rho_true = rho_list

            rel_fullK_all = [];
            rel_CM_all    = [];

            % repeat across datasets
            for sd = seeds_data

                % simulate dataset
                cfg = cfg0;
                cfg.rho = rho_true;
                out = simulate_data_dynamic(sd, trainF, cfg);

                LF  = out.LF;
                HFt = out.HF_train;

                X_L = [LF.t,  LF.s1,  LF.s2];
                y_L = LF.fL;
                X_H = [HFt.t, HFt.s1, HFt.s2];
                y_H = HFt.fH;

                % ModelInfo
                ModelInfo = struct();
                ModelInfo.X_L = X_L; ModelInfo.X_H = X_H;
                ModelInfo.y_L = y_L; ModelInfo.y_H = y_H;
                ModelInfo.jitter  = 1e-8;
                ModelInfo.nn_size = nn_size;
                ModelInfo.kernel = "RBF";
                ModelInfo.cov_type = 'RBF';
                ModelInfo.combination = 'multiplicative';
                ModelInfo.conditioning="Corr"
                ModelInfo.MeanFunction = "const";
                ModelInfo.RhoFunction  = "const";
                ModelInfo.GLSType      = "standard";
                ModelInfo.jitterM      = 1e-12;

                % hyper center
                hyp0 = zeros(11,1);
                hyp0(1)  = log(1.0);  hyp0(2)  = log(0.12);
                hyp0(3)  = log(0.6);  hyp0(4)  = log(0.12);
                hyp0(5)  = rho_true;
                hyp0(6)  = log(1e-2); hyp0(7)  = log(2e-2);
                hyp0(8)  = log(1.0);  hyp0(9)  = log(1.0);
                hyp0(10) = log(0.6);  hyp0(11) = log(1.0);

                log_idx = [1 2 3 4 6 7 8 9 10 11];

                for k = 1:n_trials_hyp
                    hyp = hyp0;
                    hyp(log_idx) = hyp(log_idx) + scale_log*randn(numel(log_idx),1);
                    hyp(5) = max(-0.99, min(0.999, rho_true + scale_rho*randn));

                    % FULL
                    try
                        NL_full = likelihood2Dsp(hyp);
                    catch
                        continue
                    end

                    % Full-K Vecchia
                    try
                        NL_fk = nlml_vecchia_fullMF(hyp);
                        rel_fullK_all(end+1,1) = abs(NL_fk - NL_full) / max(abs(NL_full),1e-12);
                    catch
                    end

                    % CM/Nested Vecchia
                    try
                        NL_cm = likelihoodVecchia_nonstat_GLS_v2(hyp);
                        rel_CM_all(end+1,1) = abs(NL_cm - NL_full) / max(abs(NL_full),1e-12);
                    catch
                    end
                end
            end

            med_fk = median(rel_fullK_all,'omitnan');
            med_cm = median(rel_CM_all,'omitnan');

            win_fk = ""; win_cm = "";
            if ~isnan(med_fk) && ~isnan(med_cm)
                if med_fk < med_cm
                    win_fk = "✓";
                elseif med_cm < med_fk
                    win_cm = "✓";
                else
                    win_fk = "="; win_cm = "=";
                end
            end

            fprintf('%6.3f | %14.3e %6s | %14.3e %6s\n', ...
                rho_true, med_fk, win_fk, med_cm, win_cm);

        end
    end
end

fprintf('\nTIP: you should see Full-K win at low rho, and CM win as rho→1 with sparse HF + small nn.\n\n');

end
