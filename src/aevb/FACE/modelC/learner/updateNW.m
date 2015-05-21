function NW = updateNW(Mu, NW)
% constant prior parameters
%   NW.nu0
%   NW.invW0, inverse of (equivalent to) NW.W0
%   NW.beta0
%   NW.mu0

    if(~iscell(Mu))
        error('the prior should observe cell data');
    end
    
    nBlocks = length(Mu);
    m = size(Mu{1},2);
    
    for blockID = 1:nBlocks
        % previous statistics
        sum1 = NW.sum1{blockID};
        sum2 = NW.sum2{blockID};
        N = NW.N;
        
        % new statistics: update sum1, sum2 and N
        mu = Mu{blockID};
        sum1 = (N*sum1+mu*mu')/(N+m);
        sum2 = (N*sum2+sum(mu,2))/(N+m);
        N = N+m;
        
        beta0 = NW.beta0{blockID};
        nu0 = NW.nu0{blockID};
        invW0 = NW.invW0{blockID};
        mu0 = NW.mu0{blockID};
        
        NW.Lambda{blockID} = ...
            invW0/N + ...
            (sum1-sum2*sum2') + ...
            beta0/(beta0+N)*(sum2-mu0)*(sum2-mu0)';
        NW.Lambda{blockID} = (1+nu0/N)*inv(NW.Lambda{blockID});
        NW.mu{blockID} = beta0/(beta0+N)*mu0 + N/(beta0+N)*sum2;
        
        NW.sum1{blockID} = sum1;
        NW.sum2{blockID} = sum2;
        
    end
    NW.N = NW.N+m;
    
end
