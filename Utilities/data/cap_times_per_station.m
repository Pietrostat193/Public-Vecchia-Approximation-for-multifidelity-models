function tbl = cap_times_per_station(tbl, capN, seed)
    if isempty(capN) || capN <= 0, return; end
    ids = unique(tbl.IDStation);
    keep = false(height(tbl),1);
    for i = 1:numel(ids)
        idx = find(tbl.IDStation == ids(i));
        if numel(idx) <= capN, keep(idx) = true; 
        else
            rng(seed); 
            % Campionamento uniforme nel tempo
            pick = round(linspace(1, numel(idx), capN));
            sub_idx = idx(pick);
            keep(sub_idx) = true;
        end
    end
    tbl = tbl(keep,:);
end