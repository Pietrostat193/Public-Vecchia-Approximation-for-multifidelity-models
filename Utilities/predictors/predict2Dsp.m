function [mean_pred] = predict2Dsp(x_star)


%global PredictionClassic
global ModelInfo

X_L = ModelInfo.X_L;
X_H = ModelInfo.X_H;
y_L = ModelInfo.y_L;
y_H = ModelInfo.y_H;
hyp = ModelInfo.hyp;

 % Low-Fidelity Temporal Parameters
 %   s_sig_LF_t = exp(hyp(1));  % Temporal signal variance for the LF model
 %   t_ell_LF = exp(hyp(2));    % Temporal length scale for the LF model
    
    % High-Fidelity Temporal Parameters
 %   s_sig_HF_t = exp(hyp(3));  % Temporal signal variance for the HF model
 %   t_ell_HF = exp(hyp(4));    % Temporal length scale for the HF model
    
    % Cross-Correlation Parameter
 rho = hyp(5);              % Cross-correlation coefficient between LF and HF
    
    % Noise Parameters
 %   eps_LF = exp(hyp(6));      % Noise variance for the LF model
 %   eps_HF = exp(hyp(7));      % Noise variance for the HF model

    % Low-Fidelity Spatial Parameters
 %   s_sig_LF_s = exp(hyp(8));  % Spatial signal variance for the LF model
 %   s_ell_LF = exp(hyp(9));    % Spatial length scale for the LF model
    
    % High-Fidelity Spatial Parameters
 %   s_sig_HF_s = exp(hyp(10)); % Spatial signal variance for the HF model
 %   s_ell_HF = exp(hyp(11));   % Spatial length scale for the HF model



%hyp=exp(hyp);
D = size(X_H,2);

y = [y_L; y_H];
L=ModelInfo.L;

%hyp(8)=1;
%hyp(10)=1;
% Define your choice for the covariance function
cov_type = ModelInfo.cov_type; % Set this to 'RBF' or 'Matern' as needed

% Initialize the covariance computations based on the specified type
switch cov_type
    case 'RBF'
       
       % Use RBF kernel (k1)
       psi1_t = rho * k1(x_star(:,1), X_L(:,1), exp(hyp(1:2)));
       psi1_s =  k1(x_star(:,2:3), X_L(:,2:3), exp(hyp(8:9)));
 
        
       %psi2_t = rho^2 * k1(x_star(:,1), X_H(:,1), exp(hyp(1:2))) + k1(x_star(:,1), X_H(:,1), exp(hyp(3:4)));
       %psi2_s = rho^2 * k1(x_star(:,2:3), X_H(:,2:3), exp(hyp(8:9))) + k1(x_star(:,2:3), X_H(:,2:3), exp(hyp(10:11)));
        
       k_t1 = k1(x_star(:,1), X_H(:,1), exp(hyp(1:2)));
       k_s1 = k1(x_star(:,2:3), X_H(:,2:3), exp(hyp(8:9)));
       k_t2 = k1(x_star(:,1), X_H(:,1), exp(hyp(3:4)));
       k_s2 = k1(x_star(:,2:3), X_H(:,2:3), exp(hyp(10:11)));
       psi2 = rho^2 * (k_t1 .* k_s1) + (k_t2 .* k_s2);

    case 'RBF_separate_rho'
        % Use RBF kernel (k1)

  
       rho_s = hyp(12); 
       psi1_t = rho * k1(x_star(:,1), X_L(:,1), exp(hyp(1:2)));
       psi1_s = rho_s * k1(x_star(:,2:3), X_L(:,2:3), exp(hyp(8:9)));
        %psi1_t = k1(x_star(:,1), X_L(:,1), exp(hyp(1:2)));
        %psi1_s = k1(x_star(:,2:3), X_L(:,2:3), exp(hyp(8:9)));
        
        psi2_t = rho^2 * k1(x_star(:,1), X_H(:,1), exp(hyp(1:2))) + k1(x_star(:,1), X_H(:,1), exp(hyp(3:4)));
        psi2_s = rho_s^2 * k1(x_star(:,2:3), X_H(:,2:3), exp(hyp(8:9))) + k1(x_star(:,2:3), X_H(:,2:3), exp(hyp(10:11)));

    case 'Matern'
        % Use Matérn kernel (k_matern)
        psi1_t = rho * k_matern(x_star(:,1), X_L(:,1), exp(hyp(1:2)));
        psi1_s = rho * k_matern(x_star(:,2:3), X_L(:,2:3), exp(hyp(8:9)));
      

        psi2_t = rho^2 * k_matern(x_star(:,1), X_H(:,1), exp(hyp(1:2))) + k_matern(x_star(:,1), X_H(:,1), exp(hyp(3:4)));
        psi2_s = rho^2 * k_matern(x_star(:,2:3), X_H(:,2:3), exp(hyp(8:9))) + k_matern(x_star(:,2:3), X_H(:,2:3), exp(hyp(10:11)));
    case 'Mix'
        % Use RBF kernel (k1)
        psi1_t = rho * k1(x_star(:,1), X_L(:,1), exp(hyp(1:2)));
        psi1_s = rho * k_matern(x_star(:,2:3), X_L(:,2:3), exp(hyp(8:9)));
        

        psi2_t = rho^2 * k1(x_star(:,1), X_H(:,1), exp(hyp(1:2))) + k1(x_star(:,1), X_H(:,1), exp(hyp(3:4)));
        psi2_s = rho^2 * k_matern(x_star(:,2:3), X_H(:,2:3), exp(hyp(8:9))) + k1(x_star(:,2:3), X_H(:,2:3), exp(hyp(10:11)));


    otherwise
        error('Invalid covariance type. Choose either "RBF" or "Matern" ');
end

combination = ModelInfo.combination;
% Resulting combined covariance terms
switch combination
    case 'additive'
psi2=psi2_t+psi2_s;
psi1 = psi1_t + psi1_s;
    case 'multiplicative'
%psi2=psi2_t.*psi2_s; 
psi1 =psi1_t.*psi1_s;
    otherwise
        error('invalid combination')
end    
% Resulting combined covariance terms
q = [psi1 psi2];

%m=hyp(12);
m=0;
% calculate prediction
mean_pred = q*(L'\(L\y))+m;
%var_star = rho^2*k(x_star, x_star, hyp(1:D+1),0) ...
%  + k(x_star, x_star, hyp(D+2:2*D+2),0) ...
%  - q*(L'\(L\q'));
%var_star = abs(diag(var_star));
end
