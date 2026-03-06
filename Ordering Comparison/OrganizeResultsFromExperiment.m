%% ============================================================
%% ORGANIZE MONTE CARLO RESULTS
%% ============================================================

% Make sure Tall exists
if ~exist('Tall','var')
    error('Tall table not found in workspace.');
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