function [mean_pred, var_pred, ci_lower, ci_upper] = predictVecchia_nonstat2(x_star)
    global ModelInfo;

    block_size = 2500;
    n_star = size(x_star, 1);

    mean_pred = zeros(n_star, 1);
    var_pred = zeros(n_star, 1);
    ci_lower = zeros(n_star, 1);
    ci_upper = zeros(n_star, 1);

    for i = 1:block_size:n_star
        idx = i:min(i + block_size - 1, n_star);
        x_block = x_star(idx, :);

        [m_blk, v_blk, ci_l_blk, ci_u_blk] = predict_block(x_block);
        mean_pred(idx) = m_blk;
        var_pred(idx) = v_blk;
        ci_lower(idx) = ci_l_blk;
        ci_upper(idx) = ci_u_blk;

        if mod(i, 2500) == 1
            fprintf("Processed prediction block %d to %d\n", idx(1), idx(end));
        end
    end
end
