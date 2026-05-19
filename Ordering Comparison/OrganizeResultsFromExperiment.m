%% ============================================================
%% ORGANIZE MONTE CARLO RESULTS
%% ============================================================

% Use Tall when available; otherwise fall back to ans from sim_vecchia_ordering_experiment_20runs().
if ~exist('Tall','var')
    if exist('ans','var') && istable(ans)
        Tall = ans;
    else
        error('Tall table not found in workspace, and ans is not a table.');
    end
end

% Aggregate across runs
G = groupsummary(Tall, {'Ordering','nn'}, ...
                 {'mean','std'}, ...
                 {'DiffAbs','DiffRel','nnzR','clipFrac_LF','clipFrac_HF'});

% Rename columns for readability
G.Properties.VariableNames = strrep(G.Properties.VariableNames,'mean_','Mean_');
G.Properties.VariableNames = strrep(G.Properties.VariableNames,'std_','Std_');

% Reorder columns nicely
G = movevars(G, {'Ordering','nn'}, 'Before', 1);

% Sort by best accuracy first
G = sortrows(G, 'Mean_DiffAbs');

%% Display final organized table
fprintf('\n================ ORGANIZED RESULTS (Mean ± Std) ================\n');
disp(G);


% Ensure Ordering is categorical (preserves clean grouping)
if ~iscategorical(G.Ordering)
    G.Ordering = categorical(G.Ordering);
end

% Sort by Ordering first, then by nn (ascending)
G_sorted = sortrows(G, {'Ordering','nn'}, {'ascend','ascend'});

% Display result
disp('============= ORGANIZED RESULTS (Grouped by Ordering, increasing nn) =============');
disp(G_sorted);