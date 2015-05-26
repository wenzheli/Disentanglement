function NW = defaultNWinit(NN, nwID)
    
    if(nargin<2)
        nwID = 2;
    end
    
    % here is a set of Normal-Sishart dist. parameters to choose from
    % col-1: beta
    % col-2: nu
    % comment: 
    %   1. have NO IDEA what is most reasonable scale of these two numbers
    %   2. if beta and nu are very large compared with mbSize, e.g. 5000, 
    %       may affect learned parameters more significantly
    nwishParams = [1 1;
        100 100;
        1 100;
        100 1];
    
    % structure variable "NW" records all information 
    %   relevant to the Normal-Wishart prior, consisting of 3 parts
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
    end
    NW.mu = zeros(NN.D2,1);
    NW.Lambda = zeros(NN.D2, NN.D2);
end
