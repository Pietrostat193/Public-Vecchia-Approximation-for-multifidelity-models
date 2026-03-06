%% TrialScript_1station.m
% Loads the two LOSO result tables (main + warp),
% normalizes schema (especially ModelInfo), concatenates safely,
% and prints a clean summary view.

clear; clc;

%% ---------- LOAD BOTH FILES ----------
load('Results_real__south_L.mat');   % expected to contain "results" (36x..)
load('results_warp.mat');            % expected to contain "results_warp" (18x..)

% --- find the main table variable ---
mainTbl = [];
if exist('results','var') && istable(results)
    mainTbl = results;
else
    % fallback: search workspace for a table that looks like the main one
    vars = who;
    for k = 1:numel(vars)
        v = eval(vars{k});
        if istable(v) && any(strcmpi(v.Properties.VariableNames,'holdout_station'))
            % likely candidate; prefer the bigger one
            if isempty(mainTbl) || height(v) > height(mainTbl)
                mainTbl = v;
            end
        end
    end
end
if isempty(mainTbl)
    error('Could not find main results table (expected variable "results").');
end

% --- find the warp table variable ---
warpTbl = [];
if exist('results_warp','var') && istable(results_warp)
    warpTbl = results_warp;
else
    % fallback: search workspace for a table that looks like the warp one
    vars = who;
    for k = 1:numel(vars)
        v = eval(vars{k});
        if istable(v) && any(strcmpi(v.Properties.VariableNames,'mu_pred_warped'))
            warpTbl = v;
            break;
        end
    end
end
if isempty(warpTbl)
    error('Could not find warp results table (expected variable "results_warp").');
end

fprintf('MAIN table size: %d x %d\n', height(mainTbl), width(mainTbl));
fprintf('WARP table size: %d x %d\n', height(warpTbl), width(warpTbl));

%% ---------- NORMALIZE & EXTRACT ----------
rows_main = localExtractNormalize(mainTbl, "results");
rows_warp = localExtractNormalize(warpTbl, "results_warp");

%% ---------- CONCATENATE SAFELY ----------
% Ensure both have the same variables (add missing as NaN/empty)
[rows_main2, rows_warp2] = alignTables(rows_main, rows_warp);

ALL = [rows_main2; rows_warp2];

%% ---------- SAFE DISPLAY (no crash if CORR missing etc.) ----------
varsWanted = ["holdout_station","modelKey","case_name","RhoFunction","RMSE","MAE","CORR","NLML"];
varsHave   = intersect(varsWanted, string(ALL.Properties.VariableNames), 'stable');

disp(ALL(:, varsHave));

%% ---------- OPTIONAL: QUICK PER-MODEL SUMMARY ----------
if all(ismember(["modelKey","RMSE","MAE"], string(ALL.Properties.VariableNames)))
    G = groupsummary(ALL, "modelKey", "mean", ["RMSE","MAE","CORR"]);
    disp(G);
end

%% ======================== LOCAL FUNCTIONS =========================

function T2 = localExtractNormalize(T, sourceName)
% Returns a normalized table with consistent columns.
% Critical fixes:
%   - ModelInfo is always a CELL (each entry is a 1x1 struct)
%   - CORR exists: if missing, computed from mu_pred & y_true when available
%   - Adds a 'modelKey' column mapping to your 4 models

    % --- standardize column name casing (MATLAB tables are case-sensitive)
    % do nothing here unless needed; we just access safely via helper.

    n = height(T);

    % ---- pull / create key columns safely ----
    holdout_station = getVarOrDefault(T, "holdout_station", nan(n,1));
    case_name       = getVarOrDefault(T, "case_name", strings(n,1));
    RhoFunction     = getVarOrDefault(T, "RhoFunction", strings(n,1));
    NLML            = getVarOrDefault(T, "NLML", nan(n,1));
    RMSE            = getVarOrDefault(T, "RMSE", nan(n,1));
    MAE             = getVarOrDefault(T, "MAE",  nan(n,1));

    % predictions / truth may be cell arrays (your tables show {100x1 double})
    mu_pred  = getVarOrDefault(T, "mu_pred", cell(n,1));
    y_true   = getVarOrDefault(T, "y_true",  cell(n,1));
    Xstar    = getVarOrDefault(T, "Xstar",   cell(n,1));

    % warp table sometimes has mu_pred_warped instead of mu_pred
    if ~hasVar(T,"mu_pred") && hasVar(T,"mu_pred_warped")
        mu_pred = T.mu_pred_warped;
    end

    % ---- normalize ModelInfo into a cell column ----
    ModelInfo = cell(n,1);
    if hasVar(T,"ModelInfo")
        MIcol = T.ModelInfo;
        for i = 1:n
            if iscell(MIcol)
                ModelInfo{i} = MIcol{i};
            else
                % struct column (non-cell) -> take row i
                ModelInfo{i} = MIcol(i);
            end

            % enforce 1x1 struct
            if isempty(ModelInfo{i})
                ModelInfo{i} = struct();
            end
        end
    else
        % no ModelInfo in table
        for i = 1:n, ModelInfo{i} = struct(); end
    end

    % ---- ensure CORR exists (compute if missing) ----
    if hasVar(T,"CORR")
        CORR = getVarOrDefault(T, "CORR", nan(n,1));
    else
        CORR = nan(n,1);
        for i = 1:n
            [yp, yt] = getPredTruth(mu_pred, y_true, i);
            if ~isempty(yp) && ~isempty(yt) && numel(yp)==numel(yt) && numel(yp) >= 2
                C = corrcoef(yp(:), yt(:));
                CORR(i) = C(1,2);
            end
        end
    end

    % ---- define modelKey (your 4 models) ----
    modelKey = strings(n,1);
    for i = 1:n
        cn = string(case_name(i));
        rf = string(RhoFunction(i));

        if contains(lower(cn), "gp-st") || contains(lower(cn), "gp st")
            modelKey(i) = "gp_st";
        elseif contains(lower(cn), "warped") || (sourceName=="results_warp")
            modelKey(i) = "mf_warp";
        elseif contains(lower(rf), "gp_scaled_emp")
            modelKey(i) = "mf_gp";
        else
            modelKey(i) = "mf_const";
        end
    end

    % ---- build normalized output table ----
    T2 = table();
    T2.holdout_station = holdout_station;
    T2.modelKey        = modelKey;
    T2.case_name        = case_name;
    T2.RhoFunction      = RhoFunction;
    T2.NLML             = NLML;
    T2.RMSE             = RMSE;
    T2.MAE              = MAE;
    T2.CORR             = CORR;
    T2.ModelInfo        = ModelInfo;  % ALWAYS cell -> avoids vertcat error
    T2.mu_pred          = mu_pred;
    T2.y_true           = y_true;
    T2.Xstar            = Xstar;
end

function [A2,B2] = alignTables(A,B)
% Make A and B have identical variables in identical order, adding missing ones.

    aNames = string(A.Properties.VariableNames);
    bNames = string(B.Properties.VariableNames);
    allNames = unique([aNames bNames],'stable');

    A2 = addMissingVars(A, allNames);
    B2 = addMissingVars(B, allNames);

    % reorder
    A2 = A2(:, allNames);
    B2 = B2(:, allNames);
end

function T2 = addMissingVars(T, allNames)
    T2 = T;
    cur = string(T.Properties.VariableNames);
    n = height(T);

    for k = 1:numel(allNames)
        v = allNames(k);
        if ~ismember(v, cur)
            % choose a sensible default type
            if v=="ModelInfo" || v=="mu_pred" || v=="y_true" || v=="Xstar"
                T2.(v) = cell(n,1);
            elseif v=="case_name" || v=="RhoFunction" || v=="modelKey"
                T2.(v) = strings(n,1);
            else
                T2.(v) = nan(n,1);
            end
        end
    end
end

function tf = hasVar(T, varname)
    tf = any(strcmp(string(T.Properties.VariableNames), string(varname)));
end

function out = getVarOrDefault(T, varname, defaultVal)
    if hasVar(T, varname)
        out = T.(varname);
    else
        out = defaultVal;
    end
end

function [yp, yt] = getPredTruth(mu_pred, y_true, i)
% Works when mu_pred/y_true are either cell arrays or numeric arrays.
    yp = []; yt = [];

    try
        if iscell(mu_pred), yp = mu_pred{i}; else, yp = mu_pred(i,:); end
    catch, yp = []; end

    try
        if iscell(y_true),  yt = y_true{i};  else, yt = y_true(i,:);  end
    catch, yt = []; end

    if isempty(yp) || isempty(yt), return; end
    if ~isnumeric(yp) || ~isnumeric(yt)
        yp = []; yt = [];
    end
end
