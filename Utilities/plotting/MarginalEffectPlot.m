% ============================================================
% Confronto distribuzioni Student-t location-scale: varia mu, sigma, nu
% 3 pannelli; in ogni pannello confronto Regime A vs Regime B.
% Plot colorate (linee con colori diversi).
% ============================================================

clear; close all; clc;

% ---- Griglia x ----
x = linspace(-10, 10, 2000);

% ---- Parametri baseline (Regime B) ----
mu0    = 0;
sigma0 = 1;
nu0    = 6;

% ---- Funzione densità Student-t location-scale: Y = mu + sigma*T, T ~ t_nu ----
dts_locscale = @(x,mu,sigma,nu) tpdf((x - mu)./sigma, nu) ./ sigma;

% ---- Colori (MATLAB default-like, ma espliciti) ----
colB = [0.00 0.45 0.74];  % blu (Regime B)
colA = [0.85 0.33 0.10];  % arancio (Regime A)

figure('Color','w','Position',[100 100 1250 380]);
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

% ============================================================
% PANNELLO 1: varia mu (shift del livello), sigma e nu fissi
% ============================================================
muA = 2;      % Regime A: livello più alto
muB = mu0;    % Regime B: baseline

yB = dts_locscale(x, muB, sigma0, nu0);
yA = dts_locscale(x, muA, sigma0, nu0);

nexttile;
plot(x, yB, 'LineWidth', 2.2, 'Color', colB); hold on;
plot(x, yA, '--', 'LineWidth', 2.2, 'Color', colA);
grid on; box on;

title('\mu varying (\sigma,\nu fixed)','Interpreter','tex');
xlabel('Value'); ylabel('Density');
legend( ...
    sprintf('Regime B: \\mu=%.1f, \\sigma=%.1f, \\nu=%.0f', muB, sigma0, nu0), ...
    sprintf('Regime A: \\mu=%.1f, \\sigma=%.1f, \\nu=%.0f', muA, sigma0, nu0), ...
    'Location','northeast');

% ============================================================
% PANNELLO 2: varia sigma (ampiezza/incertezza), mu e nu fissi
% ============================================================
sigmaA = 1.8;   % Regime A: più incertezza
sigmaB = sigma0;

yB = dts_locscale(x, mu0, sigmaB, nu0);
yA = dts_locscale(x, mu0, sigmaA, nu0);

nexttile;
plot(x, yB, 'LineWidth', 2.2, 'Color', colB); hold on;
plot(x, yA, '--', 'LineWidth', 2.2, 'Color', colA);
grid on; box on;

title('\sigma varying (\mu,\nu fixed)','Interpreter','tex');
xlabel('Value'); ylabel('Density');
legend( ...
    sprintf('Regime B: \\mu=%.1f, \\sigma=%.1f, \\nu=%.0f', mu0, sigmaB, nu0), ...
    sprintf('Regime A: \\mu=%.1f, \\sigma=%.1f, \\nu=%.0f', mu0, sigmaA, nu0), ...
    'Location','northeast');


% ============================================================
% PANNELLO 3: varia nu (peso delle code / tail risk), mu e sigma fissi
% ============================================================
nuA = 3;     % Regime A: code più pesanti
nuB = nu0;   % Regime B: baseline

yB = dts_locscale(x, mu0, sigma0, nuB);
yA = dts_locscale(x, mu0, sigma0, nuA);

nexttile;
plot(x, yB, 'LineWidth', 2.2, 'Color', colB); hold on;
plot(x, yA, '--', 'LineWidth', 2.2, 'Color', colA);
grid on; box on;

title('\nu varying (\mu,\sigma fixed)','Interpreter','tex');
xlabel('Value'); ylabel('Density');
legend( ...
    sprintf('Regime B: \\mu=%.1f, \\sigma=%.1f, \\nu=%.0f', mu0, sigma0, nuB), ...
    sprintf('Regime A: \\mu=%.1f, \\sigma=%.1f, \\nu=%.0f', mu0, sigma0, nuA), ...
    'Location','northeast');


% ---- Titolo generale ----
sgtitle('Effetti marginali su \mu (livello), \sigma (incertezza), \nu (tail risk) - Student-t location-scale');

% Nota: tpdf richiede Statistics and Machine Learning Toolbox.
% Se non lo hai, dimmelo e ti do una versione con implementazione manuale di tpdf.
