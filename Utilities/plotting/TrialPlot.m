% vero rho per stazione
sc = out.station_coords;

% --- trova automaticamente il nome della colonna con rho ---
vars = string(sc.Properties.VariableNames);

if any(vars == "rho_loc")
    rho_col = "rho_loc";
elseif any(vars == "rho_true")
    rho_col = "rho_true";
elseif any(vars == "rho")
    rho_col = "rho";
elseif any(vars == "rholoc")
    rho_col = "rholoc";
else
    error("Non trovo la colonna di rho in out.station_coords. Variabili disponibili: %s", ...
          strjoin(vars, ", "));
end

rho_station_true = sc.(rho_col);

% rho_local stimata nel likelihood (salvata da GP_scaled_empirical)
rho_local = ModelInfo.rho_local;      % n_unique_locs x 1
Xuniq     = ModelInfo.X_H_unique;     % [n_unique_locs x 2] = (s1,s2)

% mappa rho vera su Xuniq
[tf, loc] = ismember(Xuniq, [sc.s1 sc.s2], 'rows');

rho_true = nan(size(rho_local));
rho_true(tf) = rho_station_true(loc(tf));

% plot diagnostico
figure;
scatter(rho_true, rho_local, 50, 'filled'); grid on;
xlabel('rho true (simulation)'); ylabel('rho local estimated');
title('Check: rho true vs rho\_local');

% opzionale: correlazione (solo dove definito)
ok = isfinite(rho_true) & isfinite(rho_local);
if any(ok)
    r = corr(rho_true(ok), rho_local(ok));
    fprintf('Corr(rho_true, rho_local) = %.3f (n=%d)\n', r, sum(ok));
else
    disp('Nessun match tra X_H_unique e station_coords per calcolare correlazione.');
end
