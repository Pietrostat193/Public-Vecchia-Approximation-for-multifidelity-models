% repro_table4_GP3D.m
% -------------------------------------------------------------------------
% Thin wrapper to reproduce TABLE 4 of the paper — GP-3D baseline row:
%   Single-fidelity sparse GP baseline (fitrgp, SR, ARD-SE) over the
%   same 18 South-Lombardy stations used for the MFGP rows.
%
% Underlying script:
%   RealDataExperiment/GP_realDataExperiment.m
%
% Default setting: capN = 100 (time points per station).
% -------------------------------------------------------------------------
repoRoot = fileparts(mfilename('fullpath'));
addpath(genpath(repoRoot));
run(fullfile(repoRoot, 'RealDataExperiment', 'GP_realDataExperiment.m'));
