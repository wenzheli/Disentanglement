function [NN] = pretrainB_diagZ(NN, NNsetting, dataTr)
%   
    
    NW = defaultNWbInit(NN);
    nSamples = size(dataTr,2);
    numBatch = floor(nSamples/NNsetting.mbSize);

    dataH = zeros(NN.D1, nSamples);
    for firstIdx = 1:1000:nSamples
        lastIdx = min(firstIdx+999, nSamples);
        dataH(:,firstIdx:lastIdx) = 1./(1+exp(-bsxfun(@plus, NN.W1'*dataTr(:,firstIdx:lastIdx), NN.b1)));
    end
    
    quickStop=0;
    reconVar= zeros(NN.D1, NNsetting.mbSize*NNsetting.L);
    NN.ptZLoss = zeros(NNsetting.nEpoch,1);
    for epoch = 1:NNsetting.nEpoch
        if(quickStop==1)
            break;
        end
        
        index = randperm(nSamples);
        for batchIdx = 1:numBatch
            if(quickStop==1)
                break;
            end
            NN.h1 = dataH(:, index((batchIdx-1)*NNsetting.mbSize+1:batchIdx*NNsetting.mbSize));
            
            NN = ffDiagZ(NN, NNsetting, reconVar);
            NN.ptZLoss(epoch) = NN.ptZLoss(epoch)+NN.loss;
            if(rem(batchIdx,100)==1)
                fprintf(2,'epoch %d, minibatch %d, recon loss: %f\n', ...
                    epoch, batchIdx, NN.ptZLoss(epoch)/batchIdx/NNsetting.mbSize/NNsetting.L);
            end
            
            NW = updateNWb(NN.mu, NW, NN.blocks);
            NN = backDiagZ(NN, NNsetting, NW);
            
            %% update NN parameter
            NN = updateDiagZ(NN, NNsetting);
            
        end
    end
end

function NN = ffDiagZ(NN, setting, x)
    L = setting.L;
    batchSize = size(NN.h1,2);
    
    NN.mu = bsxfun(@plus, NN.W2'*NN.h1, NN.b2);
    NN.beta = bsxfun(@plus, NN.W3'*NN.h1, NN.b3);
    NN.sigma = exp(0.5*NN.beta);
    % note: covariance NN.sigma^2 = exp(NN.beta);
    
    NN.epsilon = randn(NN.D2, batchSize*L);
    NN.Sigmas = NN.epsilon*0;
    idx = (0:batchSize-1)*L;
    for i=1:L
        NN.Z(:,idx+i) = NN.mu;
        NN.Sigmas(:,idx+i) = NN.sigma;
    end
    NN.Z = NN.Z + NN.Sigmas.*NN.epsilon;
    
    sanityCheck = 0;
    if(sanityCheck)
        Z = NN.Z;
        for i=1:batchSize
            id1 = (i-1)*L+1;
            id2 = i*L;
            Z(:, id1:id2) = bsxfun(@plus, diag(NN.sigma(:,i))*NN.epsilon(:,id1:id2), NN.mu(:,i));
        end
        if(sum(sum(abs(Z-NN.Z)))>1e-10)
            error('incorrect assignment of values');
        end
    end

    NN.h2 = 1./(1+exp(-bsxfun(@plus, NN.W4'*NN.Z, NN.b4)));
    
    for i=1:setting.L
        x(:,idx+i) = NN.h1;
    end
    NN.loss = -sum(sum(x.*log(NN.h2+1e-32) + (1-x).*log(1-NN.h2+1e-32)));
    NN.delta2 = x-NN.h2;
    % Z2-->sigmoid-->h2 
    % h1-h2 := dL/dZ2
end

function NN = backDiagZ(NN, NNsetting, nw)

    L = NNsetting.L;
    batchSize = size(NN.h1,2);
    
    NN.delta1a = NN.W4*NN.delta2;
    NN.delta1b = (NN.W4*NN.delta2).*NN.epsilon;

    NN.deltaMu = NN.deltaMu*0;
    NN.deltaSigma = NN.deltaSigma*0;
    % part 1: from reconstruction error
    for i=1:batchSize
        id1 = (i-1)*NNsetting.L+1;
        id2 = i*NNsetting.L;
        NN.deltaMu(:,i) = sum(NN.delta1a(:,id1:id2),2);
        NN.deltaSigma(:,i) = sum(NN.delta1b(:,id1:id2),2);
    end
    
    % part 2: from KL divergence
    NN.deltaMu = NN.deltaMu-L*nw.Lambda*(bsxfun(@minus, NN.mu, nw.mu));
    invSigmaPrior = diag(nw.Lambda);
    NN.deltaSigma = NN.deltaSigma - L*(bsxfun(@plus, NN.sigma, invSigmaPrior) - NN.sigma.^(-1));
    
    NN.deltaBeta = NN.deltaSigma.*exp(0.5*NN.beta)/2;
    
    %% backpropagation on parameters
    NN.dW4 = NN.Z*NN.delta2'/batchSize/L + NN.momentum*NN.dW4;
    NN.dW3 = NN.h1*NN.deltaBeta'/batchSize/L + NN.momentum*NN.dW3;
    NN.dW2 = NN.h1*NN.deltaMu'/batchSize/L + NN.momentum*NN.dW2;
    
    NN.db4 = sum(NN.delta2,2)/batchSize/L + NN.momentum*NN.db4;
    NN.db3 = sum(NN.deltaBeta,2)/batchSize/L + NN.momentum*NN.db3;
    NN.db2 = sum(NN.deltaMu,2)/batchSize/L + NN.momentum*NN.db2;
    
end

function NN = updateDiagZ(NN, setting)
    if(strcmp(setting.alg, 'sgd'))
        % SGD update parameters
        NN.W4 = NN.W4 + NN.lrate.*NN.dW4; NN.b4 = NN.b4 + NN.lrate.*NN.db4;
        NN.W3 = NN.W3 + NN.lrate.*NN.dW3; NN.b3 = NN.b3 + NN.lrate.*NN.db3;
        NN.W2 = NN.W2 + NN.lrate.*NN.dW2; NN.b2 = NN.b2 + NN.lrate.*NN.db2;
    elseif(strcmp(setting.alg, 'adadelta'))
        %% ADA update of parameters and statistics
        NN.dW4E = NN.rho*NN.dW4E + (1-NN.rho)*NN.dW4.^2;
        NN.db4E = NN.rho*NN.db4E + (1-NN.rho)*NN.db4.^2;
        NN.dW3E = NN.rho*NN.dW3E + (1-NN.rho)*NN.dW3.^2;
        NN.db3E = NN.rho*NN.db3E + (1-NN.rho)*NN.db3.^2;
        NN.dW2E = NN.rho*NN.dW2E + (1-NN.rho)*NN.dW2.^2;
        NN.db2E = NN.rho*NN.db2E + (1-NN.rho)*NN.db2.^2;

        lrateW4 = sqrt(NN.deltaW4E+NN.const)./sqrt(NN.dW4E+NN.const);
        lrateb4 = sqrt(NN.deltab4E+NN.const)./sqrt(NN.db4E+NN.const);
        lrateW3 = sqrt(NN.deltaW3E+NN.const)./sqrt(NN.dW3E+NN.const);
        lrateW2 = sqrt(NN.deltaW2E+NN.const)./sqrt(NN.dW2E+NN.const);
        lrateb3 = sqrt(NN.deltab3E+NN.const)./sqrt(NN.db3E+NN.const);
        lrateb2 = sqrt(NN.deltab2E+NN.const)./sqrt(NN.db2E+NN.const);

        NN.deltaW4 = lrateW4.*NN.dW4;
        NN.deltab4 = lrateb4.*NN.db4;
        NN.deltaW3 = lrateW3.*NN.dW3;
        NN.deltab3 = lrateb3.*NN.db3;
        NN.deltaW2 = lrateW2.*NN.dW2;
        NN.deltab2 = lrateb2.*NN.db2;

        NN.deltaW4E = NN.rho*NN.deltaW4E + (1-NN.rho)*NN.deltaW4.^2;
        NN.deltab4E = NN.rho*NN.deltab4E + (1-NN.rho)*NN.deltab4.^2;
        NN.deltaW3E = NN.rho*NN.deltaW3E + (1-NN.rho)*NN.deltaW3.^2;
        NN.deltaW2E = NN.rho*NN.deltaW2E + (1-NN.rho)*NN.deltaW2.^2;
        NN.deltab3E = NN.rho*NN.deltab3E + (1-NN.rho)*NN.deltab3.^2;
        NN.deltab2E = NN.rho*NN.deltab2E + (1-NN.rho)*NN.deltab2.^2;

        NN.W4 = NN.W4+NN.deltaW4;    NN.b4 = NN.b4+NN.deltab4;
        NN.W3 = NN.W3+NN.deltaW3;    NN.b3 = NN.b3+NN.deltab3;
        NN.W2 = NN.W2+NN.deltaW2;    NN.b2 = NN.b2+NN.deltab2;

    elseif(strcmp(setting.alg, 'adam'))
    end
end
