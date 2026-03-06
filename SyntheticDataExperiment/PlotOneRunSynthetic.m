%% ===================== MULTI-LOCATION: 4 MODELS =====================

K = 6;                 % number of spatial locations to plot
roundDigits = 8;       % robust spatial grouping
rng(1);                % reproducible location choice

t_all  = out.HF_test.t(:);
s1_all = out.HF_test.s1(:);
s2_all = out.HF_test.s2(:);

% ---- robust spatial grouping ----
s1r = round(s1_all, roundDigits);
s2r = round(s2_all, roundDigits);
locKey = [s1r, s2r];

[uniqLocs, ~, locId] = unique(locKey, 'rows', 'stable');
nLoc = size(uniqLocs,1);

if nLoc < K
    K = nLoc;
end

pick = randperm(nLoc, K);

% ---- layout ----
nCols = ceil(sqrt(K));
nRows = ceil(K / nCols);

figure;
tiledlayout(nRows, nCols, 'Padding','compact', 'TileSpacing','compact');

for k = 1:K
    thisLoc = pick(k);
    idx = (locId == thisLoc);

    % sort by time
    [ts, p] = sort(t_all(idx));
    idxList = find(idx);
    idxList = idxList(p);

    y_ts       = y(idxList);
    gp1_ts     = yhat_gp1(idxList);
    classic_ts = mu_classic(idxList);
    vec_ts     = mu_vecchia(idxList);

    nexttile; hold on; grid on;

    plot(ts, y_ts, '-o', 'LineWidth', 1.6, 'MarkerSize', 4);
    plot(ts, gp1_ts, '-',  'LineWidth', 1.4);
    plot(ts, classic_ts, '-', 'LineWidth', 1.4);
    plot(ts, vec_ts, '-', 'LineWidth', 1.4);

    title(sprintf('(s1,s2)=(%.4g, %.4g)', ...
        uniqLocs(thisLoc,1), uniqLocs(thisLoc,2)));

    xlabel('t');
    ylabel('HF value');

    if k == 1
        legend('Truth','GP1','MFGP (Exact)','Vecchia', ...
               'Location','best');
    end
end

sgtitle(sprintf('HF Test Predictions (is=%d)', is), ...
        'FontWeight','bold');
