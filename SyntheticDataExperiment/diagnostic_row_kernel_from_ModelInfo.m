function diagnostic_row_kernel_from_ModelInfo(row_i)

global ModelInfo

fprintf('\n==============================\n');
fprintf('Diagnostic for row %d\n', row_i);
fprintf('==============================\n');

%% --------------------------------------------------
% 1) Build stacked locations (same as Vecchia)
%% --------------------------------------------------

X_L = ModelInfo.X_L;
X_H = ModelInfo.X_H;

locations = [X_L, zeros(size(X_L,1),1);
             X_H, ones(size(X_H,1),1)];

N = size(locations,1);

%% --------------------------------------------------
% 2) Apply permutation if used
%% --------------------------------------------------

if isfield(ModelInfo,'perm') && ~isempty(ModelInfo.perm)
    perm = ModelInfo.perm;
    locations = locations(perm,:);
end

%% --------------------------------------------------
% 3) Get neighbors (use stored vecchia indices)
%% --------------------------------------------------

nn = ModelInfo.nn_size;

if row_i == 1
    disp('First row — no neighbors.');
    return
end

prev = 1:(row_i-1);

% geometric selection like Vecchia
xi = locations(row_i,1:3);
Xprev = locations(prev,1:3);

dt = (Xprev(:,1)-xi(1)).^2;
dx = (Xprev(:,2)-xi(2)).^2;
dy = (Xprev(:,3)-xi(3)).^2;

score = dt + dx + dy;

[~,ord] = mink(score,min(nn,length(score)));
n_ind = prev(ord);

Xnbr = locations(n_ind,:);

fprintf('Neighbors selected: %d\n', length(n_ind));

%% --------------------------------------------------
% 4) Extract hypers
%% --------------------------------------------------

hyp = ModelInfo.hyp_current;

rho = hyp(5);

s_sig_LF_t = exp(hyp(1));  t_ell_LF   = exp(hyp(2));
s_sig_HF_t = exp(hyp(3));  t_ell_HF   = exp(hyp(4));
s_sig_LF_s = exp(hyp(8));  s_ell_LF   = exp(hyp(9));
s_sig_HF_s = exp(hyp(10)); s_ell_HF   = exp(hyp(11));

%% --------------------------------------------------
% 5) Build base kernels
%% --------------------------------------------------

t = Xnbr(:,1);
s = Xnbr(:,2:3);

Kt_L = k1(t,t,[s_sig_LF_t,t_ell_LF]);
Ks_L = k1(s,s,[s_sig_LF_s,s_ell_LF]);
K_L  = Kt_L .* Ks_L;

Kt_H = k1(t,t,[s_sig_HF_t,t_ell_HF]);
Ks_H = k1(s,s,[s_sig_HF_s,s_ell_HF]);
K_H  = Kt_H .* Ks_H;

f = Xnbr(:,4);

F_LL = (f==0) * (f'==0);
F_LH = (f==0) * (f'==1);
F_HL = (f==1) * (f'==0);
F_HH = (f==1) * (f'==1);




%% --------------------------------------------------
% 6) Assemble K_nn
%% --------------------------------------------------

K_nn = F_LL .* K_L ...
     + F_LH .* (rho*K_L) ...
     + F_HL .* (rho*K_L) ...
     + F_HH .* (rho^2*K_L + K_H);


object=struct();

object.F_LL=F_LL;
object.F_HH=F_HH;
object.F_HL=F_HL;
object.F_LH=F_LH;
object.K_L=K_L;
object.K_H=K_H;
object.K_nn=K_nn;
object.Xnbr=Xnbr;
object.f=f;


ModelInfo.object=object;


%% --------------------------------------------------
% 7) Visualization
%% --------------------------------------------------

figure('Color','w','Position',[100 100 1200 700]);

subplot(3,3,1)
imagesc(K_L); colorbar; axis square
title('K_L')

subplot(3,3,2)
imagesc(K_H); colorbar; axis square
title('K_H')

subplot(3,3,3)
imagesc(K_nn); colorbar; axis square
title('Final K_{nn}')

subplot(3,3,4)
imagesc(F_LL); axis square
title('F_{LL}')

subplot(3,3,5)
imagesc(F_LH); axis square
title('F_{LH}')

subplot(3,3,6)
imagesc(F_HL); axis square
title('F_{HL}')

subplot(3,3,7)
imagesc(F_HH); axis square
title('F_{HH}')

subplot(3,3,8)
imagesc(F_LL .* K_L); colorbar; axis square
title('LL contribution')

subplot(3,3,9)
imagesc(F_HH .* (rho^2*K_L + K_H)); colorbar; axis square
title('HH contribution')

sgtitle(sprintf('Row %d — #LF=%d, #HF=%d', ...
    row_i, sum(f==0), sum(f==1)));

fprintf('LF neighbors: %d\n', sum(f==0));
fprintf('HF neighbors: %d\n', sum(f==1));

end