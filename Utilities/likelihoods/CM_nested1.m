function [A, D, DI, Z21, Z1] = CM_nested1(X_L, X_H, rho_smoothness, eps_LF, eps_HF, NonStat, y_L, y_H, varargin)
    % varargin allows optional arguments like scale_rho
    %scale_rho = []; % Default value for scale_rho
    global ModelInfo
    % Check if scale_rho is provided
   % if strcmp(NonStat, "T") && ~isempty(varargin)
    %    scale_rho = varargin{1};
    %elseif strcmp(NonStat, "T") && isempty(varargin)
    %    error('scale_rho is required when NonStat == "T".');
    %end

    % Define dimensions
    N_L = size(X_L, 1); % Number of low fidelity (LF) points
    N_H = size(X_H, 1); % Number of high fidelity (HF) points

    % Define sparse identity matrix for Z1 (N_L x N_L)
    Z1 = speye(N_L);

    % Preallocate Z21 as a sparse matrix
    Z21 = sparse(N_H, N_L);

    % Find indices where high fidelity points match low fidelity points
    [~, idx] = ismember(X_H, X_L, 'rows');

    % Handle NonStat cases
   % if NonStat == "T"
   %     [rho_values, F, ~, spatial_idx] = calculateRhoSpline2(X_H(:, 1:3), X_L, y_H, y_L, rho_smoothness);
 
   % end

    % Populate Z21 based on the rho values
    if NonStat == "F"
        % Use a single rho value for all matches
        for i = 1:N_H
            rho = rho_smoothness;
           
            if idx(i) > 0
                Z21(i, idx(i)) = rho;
            end
            rho_values="It is stationary case rhos are all equal";
        end
    else
        % Use an array of rho values for each match
        for i = 1:N_H
            rho_F = rho_smoothness;
            if idx(i) > 0
                Z21(i, idx(i)) = (rho_F(i));
            end
        end
    end

    % Create sparse identity matrix for high fidelity points
    In = speye(N_H);
    Zero = sparse(N_L, N_H); % Sparse zero matrix of appropriate size

    % Construct block matrix A as a sparse matrix
    A_top = [Z1, Zero];
    A_bottom = [Z21, In];
    A = [A_top; A_bottom];

    % Add a small jitter to avoid issues with matrix inversion
    jitter = 1e-6;
    eps_LF = eps_LF + jitter;
    eps_HF = eps_HF + jitter;

    % Create D and DI matrices using sparse diagonal matrices
    upper_diag = repmat(eps_LF, N_L, 1);
    lower_diag = repmat(eps_HF, N_H, 1);
    diag_values = [upper_diag; lower_diag];

    % Construct D as a sparse diagonal matrix
    D = spdiags(diag_values, 0, N_L + N_H, N_L + N_H);

    % Construct DI as the inverse of D (also sparse)
    diag_inv_values = 1 ./ diag_values;
    DI = spdiags(diag_inv_values, 0, N_L + N_H, N_L + N_H);
end
