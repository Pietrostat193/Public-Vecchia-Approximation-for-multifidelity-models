%% MAIN EXPERIMENT: MFGP Benchmarking (Vecchia v3 + Warping + Multi-Config)
% - Uses old ResultsHistory (if available) as warmstart
% - No paramBank
% - Saves into ResultsHistory744 and all_metrics744
% - Does NOT overwrite old ResultsHistory

%clear; clc;
global ModelInfo

%% --- CONFIGURAZIONE INIZIALE ---
capN     = 100;
dataFile = "C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\Projects\6.VecchiaApproximation\3d_SpaceTimeVecchia\1_South_lombardy\South_Lombardy_sorted_data.mat";

% Load data FIRST
S    = load(dataFile);
data = S.sorted_data;

hold_id_list = unique(data.IDStation);

% ---- New output containers ----
ResultsHistory744 = struct();
all_metrics744    = table();

% Check if old ResultsHistory exists (for warmstart)
hasOldHistory = exist('ResultsHistory','var') == 1;

% Log file
diary('Experiment_Log.txt');

% Warping toolbox path
addpath("C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\Projects\5_Warping\WMFGP-main (1)\WMFGP-main")
addpath("addpath 'C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\New\3D-Example\3D-Example\Utilities'")
%% --- CONFIGURAZIONI ---
configs = {
    'Const_RhoC',   'fixed',    'constant',            false;
    'Adap_RhoC',    'adaptive', 'constant',            false;
    'Const_W_RhoC', 'fixed',    'constant',            true;
    'Adap_W_RhoC',  'adaptive', 'constant',            true;
    'Const_RhoA',   'fixed',    'GP_scaled_empirical', false;
    'Adap_RhoA',    'adaptive', 'GP_scaled_empirical', false
};

%% ================= MAIN LOOP OVER STATIONS =================
for s_idx = 1:length(hold_id_list)

    hold_id = hold_id_list(s_idx);
    s_name  = sprintf('Station_%d', hold_id);

    fprintf('\n==========================================================\n');
    fprintf(' PROCESSING STATION: %d (%d of %d)\n', hold_id, s_idx, length(hold_id_list));
    fprintf('==========================================================\n');

    %% 1) SPLIT DATA
    is_hold = (data.IDStation == hold_id);
    if ~any(is_hold)
        fprintf('Stazione %d non trovata, salto.\n', hold_id);
        continue;
    end

    train_tbl = cap_times_per_station(data(~is_hold,:), capN, 42);
    test_tbl  = cap_times_per_station(data(is_hold,:),  capN, 42);

    X_L = [double(train_tbl.Time), double(train_tbl.Lat_LF), double(train_tbl.Lon_LF)];
    y_L = double(train_tbl.Wind_speed);

    X_H = [double(train_tbl.Time), double(train_tbl.Lat_LF), double(train_tbl.Lon_LF)];
    y_H = double(train_tbl.ws);

    Xstar  = [double(test_tbl.Time), double(test_tbl.Lat_LF), double(test_tbl.Lon_LF)];
    y_true = double(test_tbl.ws);

    %% 2) BASE ModelInfo
    ModelInfo = struct();
    ModelInfo.X_L = X_L; ModelInfo.y_L = y_L;
    ModelInfo.X_H = X_H; ModelInfo.y_H = y_H;

    ModelInfo.nn_size      = 25;
    ModelInfo.kernel       = "RBF";
    ModelInfo.jitter       = 1e-5;
    ModelInfo.MeanFunction = "zero";
    ModelInfo.conditioning = "Corr";
    ModelInfo.cov_type     = "RBF";

    %% 3) Precompute Vecchia indices (once per station)
    fprintf('Fase 1: Pre-calcolo indici Vecchia...\n');
    resL = vecchia_approx_space_time_corr_fast1(X_L,[1,1],[1,1],ModelInfo.nn_size,1e-6,"RBF",10,1,1,[]);
    resH = vecchia_approx_space_time_corr_fast1(X_H,[1,1],[1,1],ModelInfo.nn_size,1e-6,"RBF",10,1,1,[]);
    ModelInfo.idxL_precomputed = extract_vecchia_indices(resL.B, ModelInfo.nn_size);
    ModelInfo.idxH_precomputed = extract_vecchia_indices(resH.B, ModelInfo.nn_size);

    %% ================= LOOP CONFIGURATIONS =================
    for c = 1:size(configs,1)

        conf_tag = configs{c,1};
        ModelInfo.GLSType     = configs{c,2};
        ModelInfo.RhoFunction = configs{c,3};
        use_warp              = configs{c,4};

        fprintf('\n--- Config: %s ---\n', conf_tag);

        %% Warping
        if use_warp
            kernelW = "Tria";
            [~, yH_norm, bgk_H, ~] = KCDF_Estim(y_H, kernelW);
            [~, yL_norm, ~, ~]     = KCDF_Estim(y_L, kernelW);

            ModelInfo.y_H = yH_norm;
            ModelInfo.y_L = yL_norm;

            lookup_y_H = Gen_Lookup(y_H, bgk_H, kernelW);
        else
            ModelInfo.y_H = y_H;
            ModelInfo.y_L = y_L;
        end

        %% ================= OPTIMIZATION =================
        n_params = 11 + 3*(ModelInfo.RhoFunction == "GP_scaled_empirical");

        % Default random start
        hyp0 = rand(n_params,1);

        % If old ResultsHistory exists, use it
        if hasOldHistory
            try
                oldHyp = ResultsHistory.(s_name).(conf_tag).hyp(:);

                if numel(oldHyp) == n_params && all(isfinite(oldHyp))
                    hyp0 = oldHyp;
                    fprintf('Warmstart from old ResultsHistory.\n');
                else
                    fprintf('Old hyp wrong dimension → random start.\n');
                end
            catch
                fprintf('No previous hyp found → random start.\n');
            end
        end

        opts_opt = optimoptions('fminunc',...
            'Algorithm','quasi-newton',...
            'Display','iter',...
            'MaxIterations',50);

        [hyp_hat, fval] = fminunc(@likelihoodVecchia_nonstat_GLS_v3, hyp0, opts_opt);

        %% ================= PREDICTION =================
        likelihoodVecchia_nonstat_GLS_v3(hyp_hat);

        if strcmp(ModelInfo.GLSType,"adaptive")
            [mu_p, s2_p] = predict_calibratedCM3_AdaptiveGLS_v4(Xstar, ModelInfo);
        else
            [mu_p, s2_p] = predict_calibratedCM3_fixed(Xstar, ModelInfo);
        end

        %% ================= INVERSE WARP + METRICS =================
        if use_warp
            y_pred = Kernel_invNS(mu_p, lookup_y_H);
            u_p    = Kernel_invNS(mu_p + 1.96*sqrt(s2_p), lookup_y_H);
            l_p    = Kernel_invNS(mu_p - 1.96*sqrt(s2_p), lookup_y_H);
        else
            y_pred = mu_p;
            u_p    = mu_p + 1.96*sqrt(s2_p);
            l_p    = mu_p - 1.96*sqrt(s2_p);
        end

        rmse = sqrt(mean((y_true - y_pred).^2));
        mae  = mean(abs(y_true - y_pred));
        cor  = corr(y_true, y_pred);
        picp = mean((y_true >= l_p) & (y_true <= u_p)) * 100;

        fprintf('RMSE: %.4f | MAE: %.4f | PICP: %.2f%% | NLML: %.2f\n', ...
                rmse, mae, picp, fval);

        %% ================= SAVE =================
        ResultsHistory744.(s_name).(conf_tag).y_true = y_true;
        ResultsHistory744.(s_name).(conf_tag).y_pred = y_pred;
        ResultsHistory744.(s_name).(conf_tag).CI_up  = u_p;
        ResultsHistory744.(s_name).(conf_tag).CI_low = l_p;
        ResultsHistory744.(s_name).(conf_tag).hyp    = hyp_hat;

        res_row = table(hold_id,{conf_tag},rmse,mae,cor,picp,fval,...
            'VariableNames',{'Station','Config','RMSE','MAE','Corr','PICP_95','NLML'});

        all_metrics744 = [all_metrics744; res_row];
    end

    %% Save after each station
    save('Experiment_Full_Results_744.mat',...
         'ResultsHistory744','all_metrics744','-v7.3');
end

diary off;
fprintf('\n>>> ESPERIMENTO COMPLETATO. Salvato in Experiment_Full_Results_744.mat\n');

%% ================= HELPER FUNCTION =================
function idx_mat = extract_vecchia_indices(B, nn)
    n = size(B,1);
    idx_mat = zeros(n,nn);
    for i = 2:n
        cols = find(B(i,1:i-1));
        if ~isempty(cols)
            len = min(length(cols),nn);
            idx_mat(i,1:len) = cols(end-len+1:end);
        end
    end
end
