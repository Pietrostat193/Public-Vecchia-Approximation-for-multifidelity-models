function NLML = nlml_vecchia_fullMF(hyp)

global ModelInfo

X_L = ModelInfo.X_L;
X_H = ModelInfo.X_H;
y_L = ModelInfo.y_L;
y_H = ModelInfo.y_H;
nn_size = ModelInfo.nn_size;

locations = [X_L, zeros(size(X_L,1),1); ...
             X_H, ones(size(X_H,1),1)];

y = [y_L; y_H];
N = numel(y);

ModelInfo.hyp_current = hyp;

result = vecchia_fullMF(locations, nn_size);

B  = result.B;
Di = result.Di;

Ki = B' * Di * B;
log_det_K = -sum(log(diag(Di)));

ModelInfo.SIy = Ki * y;
ModelInfo.log_det_K = log_det_K;
ModelInfo.y_tilde = y;

NLML = 0.5 * (y' * (Ki * y)) + 0.5 * log_det_K + 0.5 * N * log(2 * pi);

end

function result = vecchia_fullMF(locations, nn)

global ModelInfo

[n, ~] = size(locations);
nn = min(nn, n - 1);
eps_val = 1e-8;

var_self = diag(k_space_time_fullmf(locations, locations));

if isfield(ModelInfo, 'conditioning') && ~isempty(ModelInfo.conditioning)
    conditioning = string(ModelInfo.conditioning);
else
    conditioning = "MinMax";
end
useCorr = (conditioning == "Corr");

B_rows = cell(n,1);
B_cols = cell(n,1);
B_vals = cell(n,1);
Di_vals = zeros(n,1);

for i = 1:n

    if i == 1
        Di_vals(1) = 1 / var_self(1);
        continue
    end

    prev = 1:(i - 1);

    if useCorr
        % Neighbor selection by absolute correlation with point i, using
        % the same kernel that defines the model (fidelity column included).
        K_i_prev = k_space_time_fullmf(locations(prev,:), locations(i,:));
        denom = sqrt(max(var_self(prev) * var_self(i), eps_val));
        corr_score = abs(K_i_prev(:)) ./ denom;

        % Largest correlation = best neighbor -> sort descending.
        [~, ord] = maxk(corr_score, min(nn, length(corr_score)));
    else
        xi = locations(i,1:3);
        Xprev = locations(prev,1:3);

        dt = (Xprev(:,1) - xi(1)).^2;
        dx = (Xprev(:,2) - xi(2)).^2;
        dy = (Xprev(:,3) - xi(3)).^2;

        score = dt + dx + dy;

        [~, ord] = mink(score, min(nn, length(score)));
    end
    n_ind = prev(ord);

    if ~isempty(n_ind)
        Xnbr = locations(n_ind,:);
        K_nn = k_space_time_fullmf(Xnbr, Xnbr);
        K_nn = 0.5 * (K_nn + K_nn') + eps_val * eye(length(n_ind));

        K_i_n = k_space_time_fullmf(Xnbr, locations(i,:));

        Ai = K_nn \ K_i_n(:);

        Di_vals(i) = 1 / max(var_self(i) - K_i_n(:)' * Ai, eps_val);

        B_rows{i} = repmat(i, length(n_ind), 1);
        B_cols{i} = n_ind(:);
        B_vals{i} = -Ai;
    else
        Di_vals(i) = 1 / var_self(i);
    end
end

result.B = sparse(vertcat(B_rows{:}), ...
                  vertcat(B_cols{:}), ...
                  vertcat(B_vals{:}), n, n) + speye(n);

result.Di = spdiags(Di_vals, 0, n, n);

end

function K = k_space_time_fullmf(X1, X2)

global ModelInfo

if nargin < 2 || isempty(X2)
    X2 = X1;
end

hyp = ModelInfo.hyp_current;

s_sig_LF_t = exp(hyp(1));
t_ell_LF   = exp(hyp(2));
s_sig_HF_t = exp(hyp(3));
t_ell_HF   = exp(hyp(4));
rho        = hyp(5);
eps_LF     = exp(hyp(6));
eps_HF     = exp(hyp(7));
s_sig_LF_s = exp(hyp(8));
s_ell_LF   = exp(hyp(9));
s_sig_HF_s = exp(hyp(10));
s_ell_HF   = exp(hyp(11));

t1 = X1(:,1);
t2 = X2(:,1);
s1 = X1(:,2:3);
s2 = X2(:,2:3);
f1 = X1(:,4);
f2 = X2(:,4);

Kt_L = k1(t1, t2, [s_sig_LF_t, t_ell_LF]);
Ks_L = k1(s1, s2, [s_sig_LF_s, s_ell_LF]);
K_L  = Kt_L .* Ks_L;

Kt_H = k1(t1, t2, [s_sig_HF_t, t_ell_HF]);
Ks_H = k1(s1, s2, [s_sig_HF_s, s_ell_HF]);
K_H  = Kt_H .* Ks_H;

F_LL = (f1 == 0) * (f2' == 0);
F_LH = (f1 == 0) * (f2' == 1);
F_HL = (f1 == 1) * (f2' == 0);
F_HH = (f1 == 1) * (f2' == 1);

K = F_LL .* K_L ...
  + F_LH .* (rho * K_L) ...
  + F_HL .* (rho * K_L) ...
  + F_HH .* (rho^2 * K_L + K_H);

if size(K,1) == size(K,2) && isequal(X1, X2)
    K = K + diag((f1 == 0) * eps_LF + (f1 == 1) * eps_HF);
end

end
