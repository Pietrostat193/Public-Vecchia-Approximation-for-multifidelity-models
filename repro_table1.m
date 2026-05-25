% repro_table1.m
% -------------------------------------------------------------------------
% Thin wrapper to reproduce TABLE 1 of the paper:
%   Uncertainty propagation analysis for subcomponents
%   (MinMax vs Corr conditioning).
%
% Underlying script (20-replication wrapper, paper-grade):
%   SyntheticDataExperiment/reviewer_decomp_vecchia_experiment_v3_20runs.m
%
% For a single-run / debugging version use:
%   SyntheticDataExperiment/reviewer_decomp_vecchia_experiment_v3.m
% -------------------------------------------------------------------------
repoRoot = fileparts(mfilename('fullpath'));
addpath(genpath(repoRoot));
run(fullfile(repoRoot, 'SyntheticDataExperiment', ...
    'reviewer_decomp_vecchia_experiment_v3_20runs.m'));
