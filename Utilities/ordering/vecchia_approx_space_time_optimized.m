function [result] = vecchia_approx_space_time_optimized(locations, hyp_s, hyp_t, nn, eps, kernel)
    % Optimized Vecchia approximation with space-time separable covariance 
    % and partial distance calculation for neighbor selection
    %
    % Inputs:
    % - locations: [n x 3] matrix with [time, x, y] columns
    % - hyp_s, hyp_t: Hyperparameters for spatial and temporal covariance
    % - nn: Number of nearest neighbors
    % - eps: Jitter for numerical stability
    % - kernel: either "RBF" or "Matern";
    %
    % Output:
    % - result: Struct with Di (sparse diagonal matrix) and B (sparse matrix)
    
    if nargin < 4
        nn = 15;
    end
    if nargin < 5
        eps = 1e-10;
    end

    [n, ~] = size(locations);
    if nn > n - 1
        error('Number of neighbors (nn) must be less than the number of points (n).');
    end

    % Preallocate arrays to store sparse matrix entries for B
    B_rows = [];
    B_cols = [];
    B_vals = [];
    Di_vals = zeros(n, 1);

    % Parallel loop for each location to compute neighbors and covariance
    %parfor i = 1:n
    for i = 1:n
        % Display progress occasionally
        %if mod(i, 1000) == 0
        %    fprintf('Processing COV rows  %d out of %d\n', i, n);
        %end
        
        % Initialize variables for B and Di for the current row
        if i == 1
            % First row has no neighbors
            Di_vals(i) = 1 / k_space_time(locations(1, :), locations(1, :), hyp_s, hyp_t, kernel);
            continue;
        end

        % Calculate distances from the current point to previous points
        distances = sqrt(sum((locations(1:i-1, :) - locations(i, :)).^2, 2));
        
        % Find nearest neighbors
        [~, sorted_idx] = sort(distances);
        n_ind = sorted_idx(1:min(nn, length(sorted_idx)));  % Select up to nn neighbors

        % Initialize B row and add jitter to diagonal
        if length(n_ind) == 1
            add_diag = eps;
        else
            add_diag = diag(repmat(eps, length(n_ind), 1));
        end
        
        % Compute covariance matrices for neighbors and current point
        Sigma_nind_nind = k_space_time(locations(n_ind, :), locations(n_ind, :), hyp_s, hyp_t, kernel) + add_diag;
        Sigma_i_nind = k_space_time(locations(i, :), locations(n_ind, :), hyp_s, hyp_t, kernel);
        Sigma_i_i = k_space_time(locations(i, :), locations(i, :), hyp_s, hyp_t, kernel);

        % Compute Ai and B row for the current point
        Ai = Sigma_nind_nind \ Sigma_i_nind';
        B_row_vals = -Ai';

        % Ensure n_ind and B_row_vals are column vectors for concatenation
        B_rows = [B_rows; repmat(i, length(n_ind), 1)]; % Row indices for current row
        B_cols = [B_cols; n_ind(:)];                    % Column indices (ensure n_ind is a column vector)
        B_vals = [B_vals; B_row_vals(:)];               % Values (ensure B_row_vals is a column vector)

        % Store Di values
        Di_vals(i) = 1 / (Sigma_i_i - sum(Sigma_i_nind .* Ai'));
    end

    % Construct sparse matrices outside the parfor loop
    B_mat = sparse(B_rows, B_cols, B_vals, n, n);
    B_mat = B_mat + speye(n);
    Di_mat = spdiags(Di_vals, 0, n, n);

    % Store results
    result.Di = Di_mat;
    result.B = B_mat;
end

