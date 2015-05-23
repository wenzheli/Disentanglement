function NW = updateNWb(Mu, NW, blocks)
% constant prior parameters:
%   NW.nu0
%   NW.invW0, inverse of (equivalent to) NW.W0
%   NW.beta0
%   NW.mu0
% output parameters: 
%   NW.Lambda
%   NW.mu
% data statistics: 
%   sum1
%   sum2
%   N

    if(~iscell(blocks))
        error('"blocks" should be cell array');
    end
    
    nBlocks = length(blocks);
    m = size(Mu,2);
    
    for blockID = 1:nBlocks
        % previous statistics
        sum1 = NW.sum1{blockID};
        sum2 = NW.sum2{blockID};
        N = NW.N;
        
        % new statistics: update sum1, sum2 and N
        mu = Mu(blocks{blockID},:);
        sum1 = (N*sum1+mu*mu')/(N+m);
        sum2 = (N*sum2+sum(mu,2))/(N+m);
        N = N+m;
        
        beta0 = NW.beta0{blockID};
        nu0 = NW.nu0{blockID};
        invW0 = NW.invW0{blockID};
        mu0 = NW.mu0{blockID};
        
        NW.Lambda(blocks{blockID}, blocks{blockID}) = ...
            invW0/N + ...
            (sum1-sum2*sum2') + ...
            beta0/(beta0+N)*(sum2-mu0)*(sum2-mu0)';
        NW.Lambda(blocks{blockID}, blocks{blockID}) = ...
            (1+nu0/N)*inv(NW.Lambda(blocks{blockID}, blocks{blockID}));
        NW.mu(blocks{blockID}) = beta0/(beta0+N)*mu0 + N/(beta0+N)*sum2;
        
        NW.sum1{blockID} = sum1;
        NW.sum2{blockID} = sum2;
        
    end
    NW.N = NW.N+m;
    
end
