function NW = defaultNWinit(NN, nwID)
    
    if(nargin<2)
        nwID = 2;
    end
    % a set of normal-wishart distribution parameters to choose from
    % col-1: beta
    % col-2: nu
    % have NO IDEA what is most reasonable scale of these two numbers
    nwishParams = [1 1;
        10 10;
        1 10;
        10 1];
    
    % NW structure records all information relevant to the Normal-Wishart
    % prior distribution, consisting of two parts
    %   part-1: prior hyperparameters
    %       {invW0, nu0, mu0, beta0}
    %   part-2: statistics of observations
    %       {sum1, sum2, N}
    %   part-3: posterior expectation of {mean, covariance}
    %       {mu, Lambda}
    %       Lambda is nu*W, i.e. the expectation of covariance 
    %       fully determined by part-1 and part-2
    
    NW.N = 0;
    for blockID = 1:NN.nBlocks
        dim = length(NN.blocks{blockID});
        % part-1: initialization
        NW.invW0{blockID} = eye(dim);
        NW.mu0{blockID} = zeros(dim,1);
        NW.beta0{blockID} = nwishParams(nwID,1);
        NW.nu0{blockID} = nwishParams(nwID,2);
        % part-2: initialization
        NW.sum1{blockID} = zeros(dim, dim);
        NW.sum2{blockID} = zeros(dim,1);
        % part-3: DECLARATION only
        NW.mu{blockID} = NW.mu0{blockID};
        NW.Lambda{blockID} = NW.nu0{blockID}*NW.invW0{blockID};
    end
    
end
