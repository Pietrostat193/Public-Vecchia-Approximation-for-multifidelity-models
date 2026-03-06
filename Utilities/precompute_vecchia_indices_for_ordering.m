function [idxL, idxH] = precompute_vecchia_indices_for_ordering(X_L, X_H, nn, kernelName)
% precompute_vecchia_indices_for_ordering
% Precomputes Vecchia neighbor index matrices (ordering-dependent).
%
% Inputs:
%   X_L, X_H : [n x d] locations (already in the desired ordering)
%   nn       : number of neighbors to keep
%   kernelName : string, e.g. "RBF"
%
% Output:
%   idxL : [nL x nn] neighbor indices for LF (each row i contains indices < i)
%   idxH : [nH x nn] neighbor indices for HF
%
% Requires on path:
%   vecchia_approx_space_time_corr_fast1 (your existing neighbor builder)

    if nargin < 4 || isempty(kernelName)
        kernelName = "RBF";
    end

    % Dummy hyperparameters just to drive neighbor search (NOT used for likelihood)
    hyp_dummy_s = [1, 1];
    hyp_dummy_t = [1, 1];

    % Build full Vecchia structure once
    resL = vecchia_approx_space_time_corr_fast1( ...
        X_L, hyp_dummy_s, hyp_dummy_t, nn, 1e-6, kernelName, 10, 1, 1, []);
    resH = vecchia_approx_space_time_corr_fast1( ...
        X_H, hyp_dummy_s, hyp_dummy_t, nn, 1e-6, kernelName, 10, 1, 1, []);

    % Extract neighbor indices from B
    idxL = extract_indices_from_B(resL.B, nn);
    idxH = extract_indices_from_B(resH.B, nn);
end

function idx_mat = extract_indices_from_B(B, nn)
% Each row i has nonzeros in columns corresponding to its neighbor set (subset of 1:i-1)
% We return a dense [n x nn] matrix of indices (0 padded).

    n = size(B,1);
    idx_mat = zeros(n, nn);

    for i = 2:n
        cols = find(B(i, 1:i-1)); % neighbor candidates for point i
        if ~isempty(cols)
            len = min(numel(cols), nn);
            idx_mat(i, 1:len) = cols(end-len+1:end); % take last len (often closest / most recent)
        end
    end
end
