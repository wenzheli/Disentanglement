function [NN, setting]= defaultNNsetting(NN)
    
    setting.L = 10;
    setting.mbSize = 50;
    NN.D1 = 1000;
    NN.D3 = 1000;
    NN.X1      = zeros(NN.D0, setting.mbSize);
    NN.h1      = zeros(NN.D1, setting.mbSize);
    NN.X2      = zeros(NN.D0, setting.mbSize*setting.L);
    NN.h2      = zeros(NN.D3, setting.mbSize*setting.L);
    NN.reconX  = zeros(NN.D0, setting.mbSize*setting.L);
    for classID = 1:NN.nClasses
%         NN.Y{classID}       = zeros(NN.sizeClasses(classID), setting.mbSize*setting.L);
        NN.predY{classID}   = zeros(NN.sizeClasses(classID), setting.mbSize*setting.L);
    end
    
    if(strcmp(NN.shape,'diag'))
        NN.D2 = NN.dimZ*NN.nBlocks;

        NN.mu      = zeros(NN.D2, setting.mbSize);
        NN.beta    = zeros(NN.D2, setting.mbSize);
        NN.sigma   = zeros(NN.D2, setting.mbSize);
        NN.Z       = zeros(NN.D2, setting.mbSize*setting.L);
    elseif(strcmp(NN.shape,'block'))
        NN.classBlock = 1;
        % by default, use the first block for view classification

        NN.D2 = NN.dimZ;

        for blockID = 1:NN.nBlocks
            NN.Mu{blockID}      = zeros(NN.D2, setting.mbSize);
            NN.Beta{blockID}	= zeros(NN.D2, setting.mbSize);
            NN.AM{blockID}      = zeros(NN.D2^2, setting.mbSize);
            NN.AT{blockID}      = zeros(NN.D2, NN.D2, setting.mbSize);
            NN.Z{blockID}       = zeros(NN.D2, setting.mbSize*setting.L);
            NN.blocks{blockID}  = (blockID-1)*NN.D2+1:blockID*NN.D2;
        end
    elseif(strcmp(NN.shape,'full'))
        NN.D2 = NN.dimZ*NN.nBlocks;

        NN.Mu      = zeros(NN.D2, setting.mbSize);
        NN.Beta	   = zeros(NN.D2, setting.mbSize);
        NN.AM      = zeros(NN.D2^2, setting.mbSize);
        NN.AT      = zeros(NN.D2, NN.D2, setting.mbSize);
        NN.Z       = zeros(NN.D2, setting.mbSize*setting.L);
    else
        error('the posterior distribution should be either diagonal or block');
    end
    
    if(strcmp(NN.alg, 'sgd'))
        setting.alg = NN.alg;
        setting.sgdParams = [
            1/10000 0.9;
            1/5000 0.9;
            1/500 0.9;
            1/500 0.5;
            1/5000 0.5;
            1/10000 0.5];
        NN.lrate = setting.sgdParams(1,1);
        NN.momentum = setting.sgdParams(1,2);
    elseif(strcmp(NN.alg, 'adadelta'))    
        setting.alg = NN.alg;
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

        NN.rho = adaParams(NN.learnParam,1);    % (1-rho) is weight of current batch
        NN.const = adaParams(NN.learnParam,2);  % constant used to estimate RMS(.)
        NN.momentum = 0;
    elseif(strcmp(NN.alg, 'adam'))
    else
        error('the learning algorithm should be one of sgd, adadelta and adam');
    end
    setting.nEpoch = 30;
    setting.nPtEpoch = 20;
end
