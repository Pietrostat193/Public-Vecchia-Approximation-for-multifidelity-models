% repro_table2.m
% -------------------------------------------------------------------------
% Thin wrapper to reproduce TABLE 2 of the paper:
%   Main synthetic data simulation suite (MFGP vs baselines).
%
% Underlying script:
%   SyntheticDataExperiment/Main_syntheticDataSimulation_V4.m
% -------------------------------------------------------------------------
repoRoot = fileparts(mfilename('fullpath'));
addpath(genpath(repoRoot));
run(fullfile(repoRoot, 'SyntheticDataExperiment', ...
    'Main_syntheticDataSimulation_V4.m'));
