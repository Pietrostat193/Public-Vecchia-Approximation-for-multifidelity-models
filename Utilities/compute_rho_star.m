function rho_star = compute_rho_star(x_star, X_H, hyp, RhoFunction)
    global ModelInfo;
    hyp=ModelInfo.hyp;
    switch RhoFunction
        case "constant"
            rho_star = hyp(5);

        case "linear"
            phi_x = @(x) [ones(size(x, 1), 1), x];
            beta_rho = ModelInfo.beta_rho;
            phi_star = phi_x(x_star(:, 2:3));
            rho_star = phi_star * beta_rho;

        case "polynomial"
            lat = X_H(:, 2);
            lon = X_H(:, 3);
            lat_star = x_star(:, 2);
            lon_star = x_star(:, 3);

            % Normalize
            lat_norm = (lat - min(lat)) / (max(lat) - min(lat));
            lon_norm = (lon - min(lon)) / (max(lon) - min(lon));
            lat_norm_star = (lat_star - min(lat)) / (max(lat) - min(lat));
            lon_norm_star = (lon_star - min(lon)) / (max(lon) - min(lon));

            X_H_norm = [lat_norm, lon_norm];
            X_star_norm = [lat_norm_star, lon_norm_star];

            phi_x = @(x) [ones(size(x, 1), 1), x, x.^2];
            beta_rho = ModelInfo.beta_rho;

            phi_star = phi_x(X_star_norm);
            rho_star = abs(phi_star * beta_rho);

        case "GP_scaled"
            X_train_full = X_H(:, 2:3);
            X_test_full = x_star(:, 2:3);

            [X_train_unique, ~, idx_train] = unique(X_train_full, 'rows');
            [X_test_unique, ~, idx_test] = unique(X_test_full, 'rows');

            sigma = exp(hyp(end - 1));
            ell = exp(hyp(end));

            sq_dist = @(A, B) bsxfun(@plus, sum(A.^2, 2), sum(B.^2, 2)') - 2 * (A * B');

            K_star_unique = sigma^2 * exp(-0.5 * sq_dist(X_train_unique, X_test_unique) / ell^2);
            rho_star_unique = abs(mean(K_star_unique, 1))';
            rho_star = rho_star_unique(idx_test);

        case "GP_scaled_empirical"
            X_test = x_star(:, 2:3);
            gprModel_rho = ModelInfo.gprModel_rho;
            rho_star = predict(gprModel_rho, X_test);
            % Assuming rho_star is a numeric vector or array
% and ModelInfo.rho_H_unique is a numeric vector

if any(rho_star < 0.05)
    fprintf('Rho troppo basso rilevato. Attivazione interpolazione locale...\n');
    
    % 1. Identifichiamo le posizioni delle stazioni di training (uniche)
    [coords_train, idx_unique] = unique(ModelInfo.X_H(:,2:3), 'rows');
    rho_at_train = r(idx_unique); % Valori di rho stimati alle stazioni
    
    % 2. Calcoliamo le distanze tra il punto di test e le stazioni di training
    coords_test = Xstar(1, 2:3);
    distanze = pdist2(coords_test, coords_train);
    
    % 3. Troviamo le 2 stazioni più vicine
    [sorted_dist, sorted_idx] = sort(distanze);
    k = 2; % numero di vicini
    nearest_idx = sorted_idx(1:k);
    nearest_dist = sorted_dist(1:k);
    
    % 4. Interpolazione lineare (Inverse Distance Weighting)
    % Pesi = 1/distanza (aggiungiamo eps per evitare divisioni per zero)
    weights = 1 ./ (nearest_dist + eps); 
    weights = weights / sum(weights); % Normalizzazione
    
    rho_interpolato = sum(rho_at_train(nearest_idx) .* weights');
    
    % 5. Applichiamo la correzione
    rho_star(rho_star < soglia_minima) = rho_interpolato;
    
    fprintf('Nuovo rho stimato dai %d vicini più vicini: %.4f\n', k, rho_interpolato);
end


%if rho_star < 0.01
%    rho_star(:) = mean(ModelInfo.rho_H_unique);
%    disp("out of range smoothing of rho, replace with mean training value")
%end
ModelInfo.rho_star=rho_star;
        case "GP_custom"
            if ~isfield(ModelInfo, 'B_mean') || ~isfield(ModelInfo, 'Di_mean')
                error('GP_custom requires ModelInfo.B_mean and Di_mean from Vecchia training.');
            end

            X_train = ModelInfo.X_combined;
            y_train = [ModelInfo.y_L; ModelInfo.y_H];
            B = ModelInfo.B_mean;
            Di = ModelInfo.Di_mean;
            alpha = B' * (Di * (B * y_train));

            log_sigma = hyp(12);
            log_ell = hyp(13);
            theta = [exp(log_sigma), exp(log_ell)];

            kernel = ModelInfo.kernel;
            if strcmp(kernel, 'Matern')
                kernel_func_GP = @(x1, x2, theta) k_matern(x1, x2, theta);
            else
                kernel_func_GP = @(x1, x2, theta) k1(x1, x2, theta);
            end

            n_star = size(x_star, 1);
            batch_size = 1000;
            rho_star = zeros(n_star, 1);

            for i = 1:batch_size:n_star
                idx = i:min(i + batch_size - 1, n_star);
                K_star_block = kernel_func_GP(x_star(idx, :), X_train, theta);
                rho_star(idx) = abs(K_star_block * alpha);
            end

        otherwise
            error('Unsupported RhoFunction: %s', RhoFunction);
    end
end
