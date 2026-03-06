function K = k_space_time_v2(loc1, loc2, hyp_s, hyp_t, kernel)
    % Se loc2 è vuota o uguale a loc1, calcola solo la diagonale (vettore n x 1)
    diag_only = false;
    if nargin < 2 || isempty(loc2)
        diag_only = true;
    end

    spatial_loc1 = loc1(:, 2:3);
    temporal_loc1 = loc1(:, 1);

    if diag_only
        % Calcolo ultra-rapido della diagonale
        switch kernel
            case "RBF"
                % Per RBF stazionario: sigma_s * sigma_t
                K = (hyp_s(1) * ones(size(loc1,1), 1)) .* (hyp_t(1) * ones(size(loc1,1), 1));
            case "Matern"
                K = (hyp_s(1) * ones(size(loc1,1), 1)) .* (hyp_t(1) * ones(size(loc1,1), 1));
        end
    else
        % Calcolo accoppiato standard (Matrice)
        spatial_loc2 = loc2(:, 2:3);
        temporal_loc2 = loc2(:, 1);
        
        switch kernel
            case "RBF"
                K_s = k1(spatial_loc1, spatial_loc2, hyp_s);
                K_t = k1(temporal_loc1, temporal_loc2, hyp_t);
            case "Matern"
                K_s = k_matern(spatial_loc1, spatial_loc2, hyp_s);
                K_t = k_matern(temporal_loc1, temporal_loc2, hyp_t);
        end
        K = K_s .* K_t;
    end
end