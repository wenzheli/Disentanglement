function [param, bestEnergy]= sampleBLKmemory(Z, param, setting)
% param.muX: posterior mean of sample-specific x 
% param.mu_x: prior mean of x

    param = sample_X(Z, param);
    param = sample_CG_collapsed(Z, param);
    bestEnergy = compute_energy(Z, param);
    
    bestGs = param.G;
    bestCs = param.C;
    bestcc = param.cc;
    bestXs = param.X;
    bestmuX = param.muX;

    worseCount = 0;
    for iter = 1:setting.iterations
        param = sample_X(Z, param);
        param = sample_CG_collapsed(Z, param);
        energy = compute_energy(Z, param);

        if(energy>bestEnergy)
            bestGs = param.G;
            bestCs = param.C;
            bestcc = param.cc;
            bestXs = param.X;
            bestmuX = param.muX;
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
    param.Ns = sum(bestCs);
    param.X = bestXs;
    param.muX = bestmuX;
    
    param.XX=param.XX + sum(param.X.^2,2);  % K x 1
    param.XZ=param.XZ + param.X*Z';         % K x D
    
    [param.Mu, param.Sigma] = sample_data(param);
    param.Lambda = inv(param.Sigma);
    
    % reorder to put the dimensions from same cluster together
    % for visualization purpose
    order = sortrows([param.cc (1:length(param.cc))'],1);
 	param.order = order(:,2);
    param.SigmaSort = param.Sigma(param.order, param.order);
end

function [param] = sample_X(Z, param)

    K= size(param.C, 2);
    N = size(Z, 2);
    GC = param.G.*param.C;
    
    % calculate the mean and covariance of X variables
    lambdaX = sum(GC.*GC,1)/(param.sigma_noise^2) + 1/(param.sigma_x^2);
    muX = bsxfun(@rdivide, GC'*Z/param.sigma_noise^2, lambdaX');
    
    % sample X variables
    param.X = muX + diag(1./sqrt(lambdaX))*randn(K,N);
    param.muX = muX;
end

function [param] = sample_CG_collapsed(Z, param)
% provided the observation {Z and X}, 
%   update hidden variables {G and C}

    % step 1. calculate the posterior {mu, Lambda} for every g_{d,k}
    %               as if c_{d,k}=1
    % step 2. update C_{d,k}
    % step 3. update the G_{d,k} variables whose corresponding C_{d,k}=1
    %               note: it is safe not to update G_{d,k} for C_{d,k}=0
    %               because those G_{d,k} won't affect sampling X 
        
    D = size(Z,1);
    K = size(param.C, 2);

    xx=param.XX + sum(param.X.^2,2); % K x 1
    xz=param.XZ + param.X*Z';        % K x D

    %% sample Theta, the prior distribution of variable C
    param.theta = drchrnd(param.alpha+zeros(1,param.K)+sum(param.C), param.D);
    probC = log(param.theta);
    
    LambdaG = repmat(xx'/param.sigma_noise^2+1/param.sigma_g^2, [D, 1]);
    MuG = (xz'/param.sigma_noise^2+param.mu_g/param.sigma_g^2)./LambdaG;
    
    probC = probC - 0.5*log(LambdaG) + 0.5*LambdaG.*(MuG.^2);
    probC = exp(bsxfun(@minus, probC, max(probC,2)));
    probC = bsxfun(@rdivide, probC, sum(probC,2));

    % param.cc(d) is the cluster id of d-th dimension
    param.cc = sum(repmat(rand(D,1),[1,K])>cumsum(probC,2),2)+1;
    idx = (param.cc-1)*D+(1:D)';
    param.C = param.C*0;
    param.C(idx) = 1;
    param.probC = probC;
    
    %% sample G as if C_{d,k}=1
    % only those truly C_{d,k}=1 will be updated
    sampleG = MuG + randn(D,K)./sqrt(LambdaG);
    param.G(idx) = sampleG(idx);

end

function r = drchrnd(a,n)
% e.g. A = drchrnd([1, 1, 1, 1], 3)
%   generate A, a 3x4 random variable matrix
    p = length(a);
    r = gamrnd(repmat(a,n,1), 1, n, p);
    r = r./repmat(sum(r,2), 1, p);
end

function [muZ, SigmaZ] = sample_data(param)
% calculate {mu, Sigma} in Posterior(Z|X,G,C) distribution
    muZ = zeros(param.D,1)+param.mu_x;
    
    SigmaZ = param.sigma_noise^2*eye(param.D) + ...
        param.sigma_x^2*(param.G.*param.C)*(param.G.*param.C)';
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
