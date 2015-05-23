function [NN, setting]= defaultNNsetting(dataTr, D2, learnParamID, shape, alg, nBlocks)
    
    NN.shape = shape;   % shape of posterior covariance
    NN.alg = alg;       % parameter update algorithms
    NN.nBlocks = nBlocks;   % nBlocks in prior 
    for blockID = 1:nBlocks
        NN.blocks{blockID}  = (blockID-1)*D2+1:blockID*D2;
    end

    % notations
    %   L: for every observation x, generate L samples of Z according to Q(Z|x, MLP)
    if(strcmp(shape,'diag'))
        % notations
        %   sigma and beta: std and log(std) of Z nodes
        NN.D0 = size(dataTr,1);
        NN.D1 = 1000;
        NN.D2 = D2*nBlocks;
        NN.D3 = 1000;
        setting.L = 10;
        setting.mbSize = 50;

        NN.X1      = zeros(NN.D0, setting.mbSize);
        NN.h1      = zeros(NN.D1, setting.mbSize);
        NN.mu      = zeros(NN.D2, setting.mbSize);
        NN.beta    = zeros(NN.D2, setting.mbSize);
        NN.sigma   = zeros(NN.D2, setting.mbSize);
        NN.Z       = zeros(NN.D2, setting.mbSize*setting.L);
        NN.h2      = zeros(NN.D3, setting.mbSize*setting.L);
        NN.Y       = zeros(NN.D0, setting.mbSize*setting.L);
        NN.X2      = zeros(NN.D0, setting.mbSize*setting.L);
    elseif(strcmp(shape,'block'))
        NN.D0 = size(dataTr,1);
        NN.D1 = 1000;
        NN.D2 = D2;
        NN.D3 = 1000;
        setting.L = 10;
        setting.mbSize = 50;

        NN.X1      = zeros(NN.D0, setting.mbSize);
        NN.h1      = zeros(NN.D1, setting.mbSize);
        for blockID = 1:NN.nBlocks
            NN.Mu{blockID}      = zeros(NN.D2, setting.mbSize);
            NN.Beta{blockID}	= zeros(NN.D2, setting.mbSize);
            NN.AM{blockID}      = zeros(NN.D2^2, setting.mbSize);
            NN.AT{blockID}      = zeros(NN.D2, NN.D2, setting.mbSize);
            NN.Z{blockID}       = zeros(NN.D2, setting.mbSize*setting.L);
        end
        NN.h2      = zeros(NN.D3, setting.mbSize*setting.L);
        NN.Y       = zeros(NN.D0, setting.mbSize*setting.L);
        NN.X2      = zeros(NN.D0, setting.mbSize*setting.L);
    else
        error('the posterior distribution should be either diagonal or block');
    end
    
    if(strcmp(alg, 'sgd'))
        setting.alg = alg;
        setting.sgdParams = [
            1/1000 0.9;
            1/2000 0.9;
            1/3000 0.5];
        NN.lrate = setting.sgdParams(learnParamID,1);
        NN.momentum = setting.sgdParams(learnParamID,2);
    elseif(strcmp(alg, 'adadelta'))    
        setting.alg = alg;
        % 3x4 combination of 2 learning parameters
        % in practice, select the one with best ReconErr (or other
        %   criteria), and try more learning parameters nearby
        adaParams = [
            0.1 1e-8;
            0.1 1e-7;
            0.1 1e-6;
            0.1 1e-5;
            0.5 1e-8;
            0.5 1e-7;
            0.5 1e-6;
            0.5 1e-5;
            0.9 1e-8;
            0.9 1e-7;
            0.9 1e-6;
            0.9 1e-5];
        NN.rho = adaParams(learnParamID,1);    % (1-rho) is weight of current batch
        NN.const = adaParams(learnParamID,2);  % constant used to estimate RMS(.)
        NN.momentum = 0;
    elseif(strcmp(alg, 'adam'))
    else
        error('the learning algorithm should be one of sgd, adadelta and adam');
    end
    setting.nEpoch = 20;
end
