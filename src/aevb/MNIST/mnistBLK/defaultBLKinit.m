function [blk]=defaultBLKinit(blk, N, D)
% set up a block model with K blocks
    
    blk.D = D;
    blk.DC = zeros(1,blk.K);
    blk.N = N;
    blk.C = zeros(blk.D, blk.K);
    blk.cc = zeros(blk.D,1);
    blk.G = zeros(blk.D, blk.K);
    blk.X = zeros(blk.K, blk.N);
    
    
    % 1. generate/sample G
    blk.G=blk.mu_g+blk.sigma_g*randn(D,blk.K);
    
    % 2. generate/sample X
 	blk.X=blk.mu_x + blk.sigma_x*randn(blk.K,N);
    
    % 3.a) generate/sample Theta
    blk.theta = drchrnd(blk.alpha+zeros(1,blk.K), blk.D);

    % 3.b) generate/sample C
    r1 = cumsum(blk.theta, 2);
    r2 = rand(blk.D, 1);
    blk.cc = sum(bsxfun(@minus, r1, r2)<0,2)+1;
    blk.C((blk.cc-1)*blk.D+(1:D)') = 1;
    if(sum(abs(sum(bsxfun(@times, blk.C, 1:blk.K),2)-blk.cc))~=0)
        keyboard
    end
    
    blk.Ns = sum(blk.C);
end

function r = drchrnd(a,n)
% e.g. A = drchrnd([1, 1, 1, 1], 3)
%   generate A, a 3x4 random variable matrix
    p = length(a);
    r = gamrnd(repmat(a,n,1), 1, n, p);
    r = r./repmat(sum(r,2), 1, p);
end
