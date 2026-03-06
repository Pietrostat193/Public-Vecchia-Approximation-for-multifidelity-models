function [Tall, Tmean] = experiment_20runs()

    nRuns = 20;
    Tall = table();

    for r = 1:nRuns

        fprintf('\n================ RUN %d / %d ================\n', r, nRuns);

        T = demo_plot_H_orderings_precision_inverse(r);  % pass seed properly

        T.Run = repmat(r, height(T), 1);
        Tall = [Tall; T]; 
    end

    %% Aggregate statistics
    Tmean = groupsummary(Tall, {'Ordering','nn'}, ...
                         {'mean','std'}, ...
                         {'DiffAbs','DiffRel','nnzR','clipFrac_LF','clipFrac_HF'});

    fprintf('\n================ MEAN ACCURACY =================\n');
    disp(sortrows(Tmean,'mean_DiffAbs'));

    fprintf('\n================ MEAN COST =================\n');
    disp(sortrows(Tmean,'mean_nnzR'));

end