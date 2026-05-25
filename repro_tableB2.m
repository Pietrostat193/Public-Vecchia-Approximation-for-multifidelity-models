% repro_tableB2.m
% -------------------------------------------------------------------------
% Thin wrapper to reproduce TABLE B.2 of the paper:
%   Vecchia ordering ablation experiment (20 runs).
%
% Underlying script:
%   Ordering Comparison/sim_vecchia_ordering_experiment_20runs.m
% -------------------------------------------------------------------------
repoRoot = fileparts(mfilename('fullpath'));
addpath(genpath(repoRoot));
run(fullfile(repoRoot, 'Ordering Comparison', ...
    'sim_vecchia_ordering_experiment_20runs.m'));
