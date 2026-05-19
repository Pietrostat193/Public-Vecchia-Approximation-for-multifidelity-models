%% ============================================================
%% RUN EXPERIMENT AND ORGANIZE RESULTS
%% ============================================================

clear;
clc;

[Tall, Tmean] = sim_vecchia_ordering_experiment_20runs();
OrganizeResultsFromExperiment;