function idx_mat = extract_vecchia_indices(B, nn)
    n = size(B, 1);
    idx_mat = zeros(n, nn);
    for i = 2:n
        cols = find(B(i, 1:i-1));
        if ~isempty(cols)
            len = min(length(cols), nn);
            idx_mat(i, 1:len) = cols(end-len+1:end);
        end
    end
end