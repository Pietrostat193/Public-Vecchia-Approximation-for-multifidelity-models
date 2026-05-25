function D2 = sq_dists_scaled(X, l)
  Xs = X ./ l;
  sq = sum(Xs.^2, 2);
  D2 = sq + sq' - 2*(Xs*Xs');
  D2 = max(D2, 0);
end
