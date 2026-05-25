% repro_table4_MFGP.m
% -------------------------------------------------------------------------
% Thin wrapper to reproduce TABLE 4 of the paper — MFGP rows:
%   6 MFGP configurations (Const/Adap x {RhoC, W_RhoC, RhoA})
%   via LOSO over the 18 South-Lombardy weather stations.
%
% Underlying script:
%   RealDataExperiment/RealDataExperiment_main2.m
%
% Default setting: capN = 100 (time points per station).
% -------------------------------------------------------------------------
repoRoot = fileparts(mfilename('fullpath'));
addpath(genpath(repoRoot));
run(fullfile(repoRoot, 'RealDataExperiment', 'RealDataExperiment_main2.m'));
