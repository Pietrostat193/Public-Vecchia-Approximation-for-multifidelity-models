function conds = make_sim_conditions()
% Returns an array of cfg structs to pass into simulate_data_dynamic

k = 0; conds = struct([]);

for n_space = [6 10]
for n_time  = [20 40]
for rho     = [0.2 0.6 0.9]
for sigma_d2 = [0.2 0.8 2.0] % HF delta strength (how much LF helps)

    k = k + 1;
    conds(k).name = sprintf('Ns=%d^2 Nt=%d rho=%.1f sigma_d2=%.1f', n_space, n_time, rho, sigma_d2);
    conds(k).n_space = n_space;
    conds(k).n_time  = n_time;
    conds(k).rho     = rho;
    conds(k).sigma_d2 = sigma_d2;

    % optional knobs (leave fixed unless you want to sweep them too)
    conds(k).target_corr_time   = 0.8;
    conds(k).target_corr_spaceL = 0.72;
    conds(k).target_corr_spaceD = 0.72;

end
end
end
end
end
