function [NN] = pretrainE_fullZ(NN, NNsetting, dataTr)
%
    [BLK, BLKsetting]=defaultBLKsetting(NN.nBlocks);
    [BLK]=defaultBLKinit(BLK, NNsetting.mbSize, NN.D2);

    nSamples = size(dataTr,2);
    numBatch = floor(nSamples/NNsetting.mbSize);

    dataH = zeros(NN.D1, nSamples);
    for firstIdx = 1:1000:nSamples
        lastIdx = min(firstIdx+999, nSamples);
        dataH(:,firstIdx:lastIdx) = 1./(1+exp(-bsxfun(@times, NN.W1'*dataTr(:,firstIdx:lastIdx), NN.b1)));
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
            
            NN = ptE_ffFullZ(NN, NNsetting, reconVar);
            NN.ptZLoss(epoch) = NN.ptZLoss(epoch)+NN.loss;
            if(rem(batchIdx,100)==1)
                fprintf(2,'epoch %d, minibatch %d, recon loss: %f\n', ...
                    epoch, batchIdx, NN.ptZLoss(epoch)/batchIdx/NNsetting.mbSize/NNsetting.L);
            end
            
            [BLK, ~] = sampleBLK(NN.Z, BLK, BLKsetting);
            NN = ptE_backFullZ(NN, NNsetting, BLK);
            
            %% update NN parameter
            NN = ptE_updateFullZ(NN, NNsetting);
            
        end
    end
end

function NN = ptE_ffFullZ(NN, setting, x)
    L = setting.L;
    mbSize = setting.mbSize;
    NN.Mu = bsxfun(@plus, NN.W2'*NN.h1, NN.b2);
    NN.Beta = bsxfun(@plus, NN.W3'*NN.h1, NN.b3);
    NN.AM = exp(NN.Beta);
    NN.AT = reshape(NN.AM, [NN.D2, NN.D2, mbSize]);
    NN.epsilon = randn(NN.D2, mbSize*L);

    for i=1:mbSize
        id1 = (i-1)*L+1;
        id2 = i*L;
        NN.Z(:, id1:id2) = bsxfun(@plus, NN.AT(:,:,i)*NN.epsilon(:,id1:id2), NN.Mu(:,i));
    end

    NN.h2 = bsxfun(@plus, NN.W4'*NN.Z, NN.b4);
    NN.h2 = 1./(1+exp(-NN.h2));
    
    idx = L*(0:mbSize-1);
    for i=1:L
        x(:,idx+i) = NN.h1;
    end

    NN.loss = -sum(sum(x.*log(NN.h2+1e-32) + (1-x).*log(1-NN.h2+1e-32)));
    NN.delta2 = x-NN.h2;
end

function NN = ptE_backFullZ(NN, setting, blk)
    L = setting.L;
    mbSize = setting.mbSize;
    
    NN.delta1 = NN.W4*NN.delta2;
    NN.deltaMu = NN.deltaMu*0;
    NN.deltaAT = NN.deltaAT*0;
    for i=1:mbSize
        id1 = (i-1)*L+1;
        id2 = i*L;

        % gradients from reconstruction Log-Likelihood
        NN.deltaAT(:,:,i) = ...
            NN.delta1(:,id1:id2)*transpose(NN.epsilon(:,id1:id2))/L;
        NN.deltaMu(:,i) = ...
            sum(NN.delta1(:,id1:id2),2)/L;

        % gradients from minus KL divergence
        % w.r.t A: A^{-T} - Lambda*A
        % w.r.t mu: -Lambda*(mu - Emu)
        NN.deltaAT(:,:,i) = NN.deltaAT(:,:,i) + ...
            transpose(inv(NN.AT(:,:,i)+eye(NN.D2)*1e-32)) -... 
            blk.Lambda*NN.AT(:,:,i);
        NN.deltaMu(:,i) = NN.deltaMu(:,i) - ...
            blk.Lambda*(NN.Mu(:,i)-blk.Mu);
    end
    NN.deltaAM = reshape(NN.deltaAT,[NN.D2^2, mbSize]);
    NN.deltaBeta = NN.deltaAM.*NN.AM;

    %% backpropagation - 2: w.r.t. parameters

    NN.dW4 = NN.Z*NN.delta2'/mbSize/L;
    NN.dW3 = NN.h1*NN.deltaBeta'/mbSize;
    NN.dW2 = NN.h1*NN.deltaMu'/mbSize;

    NN.db4 = sum(NN.delta2,2)/mbSize/L;
    NN.db3 = sum(NN.deltaBeta,2)/mbSize;
    NN.db2 = sum(NN.deltaMu,2)/mbSize;
end

function NN = ptE_updateFullZ(NN, setting)
        if(strcmp(setting.alg, 'sgd'))
            % SGD update parameters
            NN.b4 = NN.b4 + NN.lrate.*NN.db4;
            NN.W4 = NN.W4 + NN.lrate.*NN.dW4; 
            NN.W3 = NN.W3 + NN.lrate.*NN.dW3; 
            NN.b3 = NN.b3 + NN.lrate.*NN.db3;
            NN.W2 = NN.W2 + NN.lrate.*NN.dW2; 
            NN.b2 = NN.b2 + NN.lrate.*NN.db2;
        elseif(strcmp(setting.alg, 'adadelta'))
            %% ADA update of parameters and statistics
            NN.db4E = NN.rho*NN.db4E + (1-NN.rho)*NN.db4.^2;
            lrateb4 = sqrt(NN.deltab4E+NN.const)./sqrt(NN.db4E+NN.const);
            NN.deltab4 = lrateb4.*NN.db4;
            NN.deltab4E= NN.rho*NN.deltab4E + (1-NN.rho)*NN.deltab4.^2;
            NN.b4 = NN.b4+NN.deltab4;
            
            NN.dW4E = NN.rho*NN.dW4E + (1-NN.rho)*NN.dW4.^2;   
            NN.dW3E = NN.rho*NN.dW3E + (1-NN.rho)*NN.dW3.^2;   
            NN.db3E = NN.rho*NN.db3E + (1-NN.rho)*NN.db3.^2;
            NN.dW2E = NN.rho*NN.dW2E + (1-NN.rho)*NN.dW2.^2;   
            NN.db2E = NN.rho*NN.db2E + (1-NN.rho)*NN.db2.^2;

            lrateW4 = sqrt(NN.deltaW4E+NN.const)./sqrt(NN.dW4E+NN.const);
            lrateW3 = sqrt(NN.deltaW3E+NN.const)./sqrt(NN.dW3E+NN.const);
            lrateW2 = sqrt(NN.deltaW2E+NN.const)./sqrt(NN.dW2E+NN.const);
            lrateb3 = sqrt(NN.deltab3E+NN.const)./sqrt(NN.db3E+NN.const);
            lrateb2 = sqrt(NN.deltab2E+NN.const)./sqrt(NN.db2E+NN.const);

            NN.deltaW4 = lrateW4.*NN.dW4; 
            NN.deltaW3 = lrateW3.*NN.dW3;
            NN.deltab3 = lrateb3.*NN.db3;
            NN.deltaW2 = lrateW2.*NN.dW2; 
            NN.deltab2 = lrateb2.*NN.db2;

            NN.deltaW4E = NN.rho*NN.deltaW4E + (1-NN.rho)*NN.deltaW4.^2;
            NN.deltaW3E = NN.rho*NN.deltaW3E + (1-NN.rho)*NN.deltaW3.^2;
            NN.deltab3E = NN.rho*NN.deltab3E + (1-NN.rho)*NN.deltab3.^2;
            NN.deltaW2E = NN.rho*NN.deltaW2E + (1-NN.rho)*NN.deltaW2.^2;
            NN.deltab2E = NN.rho*NN.deltab2E + (1-NN.rho)*NN.deltab2.^2;
                
            NN.W4 = NN.W4+NN.deltaW4;    
            NN.W3 = NN.W3+NN.deltaW3;    
            NN.b3 = NN.b3+NN.deltab3;
            NN.W2 = NN.W2+NN.deltaW2;    
            NN.b2 = NN.b2+NN.deltab2;
        end
end
