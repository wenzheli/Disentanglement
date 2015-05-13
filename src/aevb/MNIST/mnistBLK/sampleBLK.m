function param = sampleBLK(Z, param, setting)
% two to add
%   1. iteration and convergence control
%   2. generate the mean and covariance of Y ( the blk prior )

    param = sample_X(Z, param);
    param = sample_CG_collapsed(Z, param);
    bestEnergy = compute_energy(Z, param);
    
    bestGs = param.G;
    bestCs = param.C;
    bestcc = param.cc;
    bestNs = param.Ns;
    bestXs = param.X;
    
    worseCount = 0;
    for iter = 1:setting.iterations
        param = sample_X(Z, param);
        param = sample_CG_collapsed(Z, param);
        % calculate objective function
        energy = compute_energy(Z, param);
        if(energy>bestEnergy)
            bestGs = param.G;
            bestCs = param.C;
            bestcc = param.cc;
            bestNs = param.Ns;
            bestXs = param.X;
        else
            worseCount = worseCount+1;
        end
        if(worseCount>5)
            break;
        end
    end
    param.G = bestGs;
    param.C = bestCs;
    param.cc = bestcc;
    param.Ns = bestNs;
    param.X = bestXs;
    
    [param.Mu, param.Sigma] = sample_data(param);
    param.Lambda = inv(param.Sigma);
end

function [param] = sample_X(Z, param)
    Gs=param.G;
    Cs=param.C;

    K= size(Cs, 2);
    N = size(Z, 2);
    Gm = Gs.*Cs;
    tem=Gm' / param.sigma_noise^2;
    lambda_diag= ((param.sigma_noise)^(-2))*sum(Gm.*Gm,1)+param.sigma_x^(-2);

    for n=1:N
        mu = (tem*Z(:,n))./lambda_diag';
        param.X(:,n) = mu + lambda_diag'.^(-0.5).*randn(K,1);
    end
end

function [param] = sample_CG_collapsed(Z, param)
% provided the observation Z, sample hidden variables
    Cs=param.C;
    cc=param.cc;
    Ns=param.Ns;
    Gs=param.G;
    Xs=param.X;
    D = size(Z,1);
    K= size(Cs, 2);

    perm=randperm(D);
    X2cache=sum(Xs.^2,2);
    
    %% sample Theta
    % from posterior Dirichlet distribution
    % param.alpha
    % 
    param.theta = drchrnd(param.alpha+zeros(1,param.K)+sum(param.C), param.D);

    %% sample G and C
    for ind=1:D

        d=perm(ind);

        cluster = cc(d);
        Ns(cluster) = Ns(cluster) - 1;
        
        % Dirichlet Prior of variable C
        p = log(param.theta(d,:));
%         p = log(Ns.*(1/(param.alpha+D-1)));
%         p((end+1):(end+m_aux)) = log((param.alpha/m_aux)) -log(param.alpha+D-1);
        
        Gsamps =zeros(1,K);
        xz=Xs*Z(d,:)';
        
        for k = 1:K
            % 1. sample G conditional distribution
            %   following equation (1) and (2) in supplemental material
            lambda = X2cache(k)/param.sigma_noise^2+1/param.sigma_g^2;
            mu=(xz(k)/param.sigma_noise^2+param.mu_g/param.sigma_g^2)/lambda;
            
            % 2. sample C conditional distribution
            %   following equation in "clustering assignments c"
            p(k) = p(k) -.5*(log(lambda) - lambda*mu^2);
            % the following terms are constant wrt k
            % -.5*(N*log(2*pi)+N*log(param.sigma_noise^2) + Z2cache(d)/param.sigma_noise^2 ...
            %    + log(param.sigma_g^2) + param.mu_g^2/param.sigma_g^2 )
            %
            % sample G_{k} ~ post(.)
            Gsamps(k)=mu + lambda^(-.5) * randn();
        end
        p = exp(p-max(p));
        p=p/(sum(p));
        
        % sample from the conditional probabilities
        % G:
        kk = 1+sum(rand>cumsum(p));
        Gs(d,kk)=Gsamps(kk);
        Ns(kk) = Ns(kk)+1;
        % C:
        cc(d)=kk;
        Cs(d,:)=zeros(1,K);
        Cs(d,kk)=1;
    end

    param.C=Cs;
    param.cc=cc;
    param.Ns=Ns;
    param.G=Gs;
    param.X=Xs;

end

function r = drchrnd(a,n)
% e.g. A = drchrnd([1, 1, 1, 1], 3)
%   generate A, a 3x4 random variable matrix
    p = length(a);
    r = gamrnd(repmat(a,n,1), 1, n, p);
    r = r./repmat(sum(r,2), 1, p);
end

function [mu, Sigma] = sample_data(param)

% provided (mu_x, sigma_x), G and C
% calculate the mean and covariance of blk distribution
    % contribution from Gaussian noise
    Sigma = param.sigma_noise*eye(param.D);
    mu = zeros(param.D,1)+param.mu_noise;
    
    Sigma = Sigma + ...
        (param.G.*param.C)*diag(param.sigma_x.^2)*(param.G.*param.C)';
    mu = mu + sum(param.G.*param.C,2)*param.mu_x;

    % if generate samples: 
%     e= param.mu_noise+ param.sigma_noise*randn(D,N);
%     Y = (param.G.*param.C) * param.X + e;
    
end

function [energy] = compute_energy(Y, param)
% Y: DxN
% C: DxK
    [D, N] = size(Y);
    lk =  compute_likelihood_fast(Y, param);

    % prior log-likeliood of G, Gaussian
    prior_g= -0.5*(1/(param.sigma_g^2)) *(sum(sum( (param.G-repmat(param.mu_g.*ones(param.D,1),[1 param.K])).^2  ) ));
    % prior log-likelihood of X, Gaussian
    prior_x = -0.5*(1/(param.sigma_x^2)) *(sum(sum( (param.X-repmat(param.mu_x.*ones(param.K,1),[1 N])).^2  ) ));
    
    % prior log-likelihood of C, ?  double-check and compare
%     prior_c = gammaln(param.alpha) + K*log(param.alpha) +sum(gammaln(param.Ns)) - gammaln(param.alpha+D);
    prior_c = sum(gammaln(param.alpha + param.Ns)) + gammaln(param.alpha*param.K) ...
        - param.K*gammaln(param.alpha) - gammaln(param.alpha*param.K+param.D);

    energy = lk+prior_g + prior_x +prior_c;

end

function [ lh ] = compute_likelihood_fast(Y,  param)
    
    [D, N] = size(Y);
    
    Gm=param.G.*param.C;
    E=Y-Gm*param.X;
    lh=-D*N*(log(param.sigma_noise)+0.5*log(2.0*pi)) - 0.5*sum(sum(E.^2))/(param.sigma_noise^2);
    
end
