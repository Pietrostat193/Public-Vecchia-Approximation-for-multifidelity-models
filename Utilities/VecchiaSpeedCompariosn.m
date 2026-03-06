%% SCRIPT DI CONFRONTO: v2 vs v3
global ModelInfo;

% --- 1. SETUP INDICI PER V3 (Corretto) ---
fprintf('Fase 1: Estrazione indici dei vicini...\n');

% Eseguiamo la funzione originale (v2) per generare le matrici dei vicini
resL = vecchia_approx_space_time_corr_fast1(ModelInfo.X_L, [1,1], [1,1], ...
    ModelInfo.nn_size, 1e-6, ModelInfo.kernel, 10, 1, 1, []);
resH = vecchia_approx_space_time_corr_fast1(ModelInfo.X_H, [1,1], [1,1], ...
    ModelInfo.nn_size, 1e-6, ModelInfo.kernel, 10, 1, 1, []);

% Inizializzazione matrici indici
nL = size(ModelInfo.X_L, 1);
nH = size(ModelInfo.X_H, 1);
nn = ModelInfo.nn_size;

ModelInfo.idxL_precomputed = zeros(nL, nn);
ModelInfo.idxH_precomputed = zeros(nH, nn);

% Estrazione manuale (più sicura delle funzioni anonime per le matrici sparse)
for i = 2:nL
    cols = find(resL.B(i, 1:i-1));
    if ~isempty(cols)
        len = min(length(cols), nn);
        ModelInfo.idxL_precomputed(i, 1:len) = cols(end-len+1:end);
    end
end

for i = 2:nH
    cols = find(resH.B(i, 1:i-1));
    if ~isempty(cols)
        len = min(length(cols), nn);
        ModelInfo.idxH_precomputed(i, 1:len) = cols(end-len+1:end);
    end
end

% --- 2. DEFINIZIONE PARAMETRI DI TEST ---
% hyp deve essere definito nel tuo workspace
hyp=rand(11,1)
hyp_test = hyp; 

% --- 3. ESECUZIONE v2 ---
fprintf('Fase 2: Esecuzione v2 (Versione standard)...\n');
tic;
nlml_v2 = likelihoodVecchia_nonstat_GLS_v2(hyp_test);
beta_v2 = ModelInfo.beta_gls;
t_v2 = toc;

% --- 4. ESECUZIONE v3 ---
fprintf('Fase 3: Esecuzione v3 (Versione Light con pre-calcolo)...\n');
tic;
nlml_v3 = likelihoodVecchia_nonstat_GLS_v3(hyp_test);
beta_v3 = ModelInfo.beta_gls;
t_v3 = toc;

% --- 5. REPORT FINALE ---
fprintf('\n================================================\n');
fprintf('         CONFRONTO NUMERICO v2 vs v3\n');
fprintf('================================================\n');
fprintf('Valore NLML v2:        %20.10f\n', nlml_v2);
fprintf('Valore NLML v3:        %20.10f\n', nlml_v3);
fprintf('Differenza Assoluta:   %20.10e\n', abs(nlml_v2 - nlml_v3));
fprintf('------------------------------------------------\n');
fprintf('Errore Max su Beta:    %20.10e\n', max(abs(beta_v2 - beta_v3)));
fprintf('------------------------------------------------\n');
fprintf('Tempo Esecuzione v2:   %10.4f s\n', t_v2);
fprintf('Tempo Esecuzione v3:   %10.4f s\n', t_v3);
fprintf('SPEED-UP:              %10.2f x\n', t_v2 / t_v3);
fprintf('================================================\n');