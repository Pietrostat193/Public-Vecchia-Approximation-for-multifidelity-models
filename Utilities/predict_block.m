function [mean_pred, var_pred, ci_lower, ci_upper] = predict_block(x_star)
    global ModelInfo;

    % Unpack model info
    X_L = ModelInfo.X_L;
    X_H = ModelInfo.X_H;
    SIy = ModelInfo.SIy;
    y = [ModelInfo.y_L; ModelInfo.y_H];
    hyp = ModelInfo.hyp;
    RhoFunction = ModelInfo.RhoFunction;
    kernel = ModelInfo.kernel;
    MeanFunction = ModelInfo.MeanFunction;

    if strcmp(kernel, 'Matern')
        kernel_func = @(x1, x2, hyp_params) k_matern(x1, x2, hyp_params);
    else
        kernel_func = @(x1, x2, hyp_params) k1(x1, x2, hyp_params);
    end

    % === Compute rho_star ===
    rho_star = compute_rho_star(x_star, X_H, hyp, RhoFunction);
    ModelInfo.rho_star=rho_star;
    % Kernel params
    hyp_exp = exp(hyp);
    s_sig_LF_t = hyp_exp(1);  t_ell_LF = hyp_exp(2);
    s_sig_HF_t = hyp_exp(3);  t_ell_HF = hyp_exp(4);
    s_sig_LF_s = hyp_exp(8);  s_ell_LF = hyp_exp(9);
    s_sig_HF_s = hyp_exp(10); s_ell_HF = hyp_exp(11);

    % Low-fidelity kernel
    psi1_t = kernel_func(x_star(:, 1), X_L(:, 1), [s_sig_LF_t, t_ell_LF]);
    psi1_s = kernel_func(x_star(:, 2:3), X_L(:, 2:3), [s_sig_LF_s, s_ell_LF]);
    psi1 = psi1_t .* psi1_s;

    % High-fidelity kernel
    k_t1 = kernel_func(x_star(:, 1), X_H(:, 1), [s_sig_LF_t, t_ell_LF]);
    k_s1 = kernel_func(x_star(:, 2:3), X_H(:, 2:3), [s_sig_LF_s, s_ell_LF]);
    k_t2 = kernel_func(x_star(:, 1), X_H(:, 1), [s_sig_HF_t, t_ell_HF]);
    k_s2 = kernel_func(x_star(:, 2:3), X_H(:, 2:3), [s_sig_HF_s, s_ell_HF]);
    psi2 = (rho_star^2) .* (k_t1 .* k_s1) + (k_t2 .* k_s2);
    
    psi = [psi1 psi2];
    
    D=SIy(length(X_L)+1:end);
    % Mean
    switch MeanFunction
        case "zero"
            m_x = 0;
        case "constant"
            m_x = hyp(12);
        case "linear"
            weights = hyp(12:14); bias = hyp(15);
            m_x = x_star * weights + bias;
        case "GP"
            m_x = predict(ModelInfo.gprModel_mean, x_star);
        case "GP_for_H"
            m_x = predict(ModelInfo.gprModel_forH, x_star(:, 2:3));
        case "GP_res"
            m_x = predict(ModelInfo.gprModel_mean_H, x_star(:, 2:3));
    end

    % Final prediction
    if MeanFunction == "GP_res"
        mean_pred = psi * SIy + m_x;
    elseif MeanFunction == "residuals_HF"
        disp("miao")
    else
        mean_pred = psi * SIy + m_x;
    end

    % Variance
    K_inv = SIy * pinv(y);
    k_tt = kernel_func(x_star(:, 1), x_star(:, 1), [s_sig_HF_t, t_ell_HF]);
    k_ss = kernel_func(x_star(:, 2:3), x_star(:, 2:3), [s_sig_HF_s, s_ell_HF]);
    var_pred = (rho_star.^2) .* (k_tt .* k_ss) - psi * (K_inv * psi');
    var_pred = abs(diag(var_pred));

    % Confidence Intervals
    ci = 1.96 * sqrt(var_pred);
    ci_lower = mean_pred - ci;
    ci_upper = mean_pred + ci;
end
