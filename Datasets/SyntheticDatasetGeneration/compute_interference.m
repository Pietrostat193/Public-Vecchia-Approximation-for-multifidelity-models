function W_Lh = compute_interference(ModelInfo, sigma_space, ell_space, sigma_time, ell_time, r_hc)
    % Compute the spatio-temporal kernel matrix using RBF kernel
    K_D = k_space_time(ModelInfo.X_L, ModelInfo.X_L, [sigma_space, ell_space], [sigma_time, ell_time], "RBF");
    
    % Add numerical stability term
    K_D = K_D + 1e-6 * eye(size(K_D));
    
    % Generate Gaussian Process sample with 100 samples and 9 dimensions
    interference = generate_gp_sample(K_D, 100, 9);
    
    % Compute the final result by adding interference to r_hc
    W_Lh = r_hc + interference;
end
