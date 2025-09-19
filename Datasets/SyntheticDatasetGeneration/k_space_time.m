function K = k_space_time(loc1, loc2, hyp_s, hyp_t, kernel)
    % loc1, loc2 are matrices of space-time coordinates
    % hyp_s: spatial hyperparameters (e.g., variance and length scale)
    % hyp_t: temporal hyperparameters (e.g., variance and length scale)

    % Spatial and temporal coordinates
    spatial_loc1 = loc1(:, 2:3);  % Spatial (x, y)
    spatial_loc2 = loc2(:, 2:3);
    temporal_loc1 = loc1(:, 1);   % Temporal (time)
    temporal_loc2 = loc2(:, 1);

    switch kernel
        case "RBF"
    % Compute spatial and temporal covariance matrices
    K_s = k1(spatial_loc1, spatial_loc2, hyp_s); % Spatial component
    K_t = k1(temporal_loc1, temporal_loc2, hyp_t); % Temporal component
        case "Matern"
        % Compute spatial and temporal covariance matrices
    K_s = k_matern(spatial_loc1, spatial_loc2, hyp_s); % Spatial component
    K_t = k_matern(temporal_loc1, temporal_loc2, hyp_t); % Temporal component
    end
    
    % Combine the spatial and temporal components
    K = K_s .* K_t; % Element-wise product for separable covariance
end
