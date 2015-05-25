function [NN] = pretrainC_blockZ(NN, NNsetting, dataTr)
% pretrain the H1-->Z-->H2 layer, 
%   i.e. the middle dim reduction layer in modelC
%   Block-NW prior is used 

    NW = defaultNWinit(NN);

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
            
            NN = ffBlockZ(NN, NNsetting, reconVar);
            NN.ptZLoss(epoch) = NN.ptZLoss(epoch)+NN.loss;
            if(rem(batchIdx,100)==1)
                fprintf(2,'epoch %d, minibatch %d, recon loss: %f\n', ...
                    epoch, batchIdx, NN.ptZLoss(epoch)/batchIdx/NNsetting.mbSize/NNsetting.L);
            end
            
            % update NW prior 
            NW = updateNW(NN.Mu, NW);
            
            % back-propagation for derivatives --> d(ReconErr+KL)/d(param)
            NN = backBlockZ(NN, NNsetting, NW);
            
            % update NN parameter
            NN = updateBlockZ(NN, NNsetting);
            
        end
    end
end

function NN = ffBlockZ(NN, setting, x)
    L = setting.L;
    mbSize = setting.mbSize;
    for blockID = 1:NN.nBlocks
        NN.Mu{blockID} = bsxfun(@plus, NN.W2{blockID}'*NN.h1, NN.b2{blockID});
        NN.Beta{blockID} = bsxfun(@plus, NN.W3{blockID}'*NN.h1, NN.b3{blockID});
        NN.AM{blockID} = exp(NN.Beta{blockID});
        NN.AT{blockID} = reshape(NN.AM{blockID}, [NN.D2, NN.D2, mbSize]);
        NN.epsilon{blockID} = randn(NN.D2, mbSize*L);

        for i=1:mbSize
            id1 = (i-1)*L+1;
            id2 = i*L;
            NN.Z{blockID}(:, id1:id2) = bsxfun(@plus, NN.AT{blockID}(:,:,i)*NN.epsilon{blockID}(:,id1:id2), NN.Mu{blockID}(:,i));
        end
    end

    NN.h2 = bsxfun(@plus, NN.W4{1}'*NN.Z{1}, NN.b4);
    for blockID = 2:NN.nBlocks
        NN.h2 = NN.h2 + NN.W4{blockID}'*NN.Z{blockID};
    end
    NN.h2 = 1./(1+exp(-NN.h2));

    
    idx = 0:mbSize-1;
    for i=1:L
        x(:,idx*L+i) = NN.h1;
    end

    NN.loss = -sum(sum(x.*log(NN.h2+1e-32) + (1-x).*log(1-NN.h2+1e-32)));
    NN.delta2 = x-NN.h2;
end

function NN = backBlockZ(NN, setting, NW)
    L = setting.L;
    mbSize = setting.mbSize;
    
    for blockID = 1:NN.nBlocks
        NN.delta1{blockID} = NN.W4{blockID}*NN.delta2;
        NN.deltaMu{blockID} = NN.deltaMu{blockID}*0;
        NN.deltaAT{blockID} = NN.deltaAT{blockID}*0;
        for i=1:mbSize
            id1 = (i-1)*L+1;
            id2 = i*L;

            % gradients from reconstruction Log-Likelihood
            NN.deltaAT{blockID}(:,:,i) = ...
                NN.delta1{blockID}(:,id1:id2)*transpose(NN.epsilon{blockID}(:,id1:id2))/L;
            NN.deltaMu{blockID}(:,i) = ...
                sum(NN.delta1{blockID}(:,id1:id2),2)/L;

            % gradients from minus KL divergence
            % w.r.t A: A^{-T} - Lambda*A
            % w.r.t mu: -Lambda*(mu - Emu)
            NN.deltaAT{blockID}(:,:,i) = NN.deltaAT{blockID}(:,:,i) + ...
                transpose(inv(NN.AT{blockID}(:,:,i)+eye(NN.D2)*1e-32)) -... 
                NW.Lambda{blockID}*NN.AT{blockID}(:,:,i);
            NN.deltaMu{blockID}(:,i) = NN.deltaMu{blockID}(:,i) - ...
                NW.Lambda{blockID}*(NN.Mu{blockID}(:,i)-NW.mu{blockID});
        end
        NN.deltaAM{blockID} = reshape(NN.deltaAT{blockID},[NN.D2^2, mbSize]);
        NN.deltaBeta{blockID} = NN.deltaAM{blockID}.*NN.AM{blockID};
    end

    %% backpropagation - 2: w.r.t. parameters

    for blockID = 1:NN.nBlocks
        NN.dW4{blockID} = NN.Z{blockID}*NN.delta2'/mbSize/L;
        NN.dW3{blockID} = NN.h1*NN.deltaBeta{blockID}'/mbSize;
        NN.dW2{blockID} = NN.h1*NN.deltaMu{blockID}'/mbSize;
    end

    NN.db4 = sum(NN.delta2,2)/mbSize/L;
    for blockID = 1:NN.nBlocks
        NN.db3{blockID} = sum(NN.deltaBeta{blockID},2)/mbSize;
        NN.db2{blockID} = sum(NN.deltaMu{blockID},2)/mbSize;
    end
end

function NN = updateBlockZ(NN, setting)
    if(strcmp(NN.shape, 'diag'))
        if(strcmp(setting.alg, 'sgd'))
            % SGD update parameters
            NN.W4 = NN.W4 + NN.lrate.*NN.dW4; NN.b4 = NN.b4 + NN.lrate.*NN.db4;
            NN.W3 = NN.W3 + NN.lrate.*NN.dW3; NN.b3 = NN.b3 + NN.lrate.*NN.db3;
            NN.W2 = NN.W2 + NN.lrate.*NN.dW2; NN.b2 = NN.b2 + NN.lrate.*NN.db2;
        elseif(strcmp(setting.alg, 'adadelta'))
            %% ADA update of parameters and statistics

            NN.db4E = NN.rho*NN.db4E + (1-NN.rho)*NN.db4.^2;
            lrateb4 = sqrt(NN.deltab4E+NN.const)./sqrt(NN.db4E+NN.const);
            NN.deltab4 = lrateb4.*NN.db4;
            NN.deltab4E= NN.rho*NN.deltab4E + (1-NN.rho)*NN.deltab4.^2;

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
            NN.deltaW2E = NN.rho*NN.deltaW2E + (1-NN.rho)*NN.deltaW2.^2;

            NN.deltab3E = NN.rho*NN.deltab3E + (1-NN.rho)*NN.deltab3.^2;
            NN.deltab2E = NN.rho*NN.deltab2E + (1-NN.rho)*NN.deltab2.^2;

            NN.W4 = NN.W4+NN.deltaW4;    NN.b4 = NN.b4+NN.deltab4;
            NN.W3 = NN.W3+NN.deltaW3;    NN.b3 = NN.b3+NN.deltab3;
            NN.W2 = NN.W2+NN.deltaW2;    NN.b2 = NN.b2+NN.deltab2;
            
        elseif(strcmp(setting.alg, 'adam'))
        end
    elseif(strcmp(NN.shape,'block'))
        if(strcmp(setting.alg, 'sgd'))
            % SGD update parameters
            NN.b4 = NN.b4 + NN.lrate.*NN.db4;
            for blockID = 1:NN.nBlocks
                NN.W4{blockID} = NN.W4{blockID} + NN.lrate.*NN.dW4{blockID}; 
                NN.W3{blockID} = NN.W3{blockID} + NN.lrate.*NN.dW3{blockID}; 
                NN.b3{blockID} = NN.b3{blockID} + NN.lrate.*NN.db3{blockID};
                NN.W2{blockID} = NN.W2{blockID} + NN.lrate.*NN.dW2{blockID}; 
                NN.b2{blockID} = NN.b2{blockID} + NN.lrate.*NN.db2{blockID};
            end
        elseif(strcmp(setting.alg, 'adadelta'))
            %% ADA update of parameters and statistics
            NN.db4E = NN.rho*NN.db4E + (1-NN.rho)*NN.db4.^2;
            lrateb4 = sqrt(NN.deltab4E+NN.const)./sqrt(NN.db4E+NN.const);
            NN.deltab4 = lrateb4.*NN.db4;
            NN.deltab4E= NN.rho*NN.deltab4E + (1-NN.rho)*NN.deltab4.^2;
            NN.b4 = NN.b4+NN.deltab4;
            
            for blockID = 1:NN.nBlocks
                NN.dW4E{blockID} = NN.rho*NN.dW4E{blockID} + (1-NN.rho)*NN.dW4{blockID}.^2;   
                NN.dW3E{blockID} = NN.rho*NN.dW3E{blockID} + (1-NN.rho)*NN.dW3{blockID}.^2;   
                NN.db3E{blockID} = NN.rho*NN.db3E{blockID} + (1-NN.rho)*NN.db3{blockID}.^2;
                NN.dW2E{blockID} = NN.rho*NN.dW2E{blockID} + (1-NN.rho)*NN.dW2{blockID}.^2;   
                NN.db2E{blockID} = NN.rho*NN.db2E{blockID} + (1-NN.rho)*NN.db2{blockID}.^2;

                lrateW4 = sqrt(NN.deltaW4E{blockID}+NN.const)./sqrt(NN.dW4E{blockID}+NN.const);
                lrateW3 = sqrt(NN.deltaW3E{blockID}+NN.const)./sqrt(NN.dW3E{blockID}+NN.const);
                lrateW2 = sqrt(NN.deltaW2E{blockID}+NN.const)./sqrt(NN.dW2E{blockID}+NN.const);
                lrateb3 = sqrt(NN.deltab3E{blockID}+NN.const)./sqrt(NN.db3E{blockID}+NN.const);
                lrateb2 = sqrt(NN.deltab2E{blockID}+NN.const)./sqrt(NN.db2E{blockID}+NN.const);

                NN.deltaW4{blockID} = lrateW4.*NN.dW4{blockID}; 
                NN.deltaW3{blockID} = lrateW3.*NN.dW3{blockID};
                NN.deltab3{blockID} = lrateb3.*NN.db3{blockID};
                NN.deltaW2{blockID} = lrateW2.*NN.dW2{blockID}; 
                NN.deltab2{blockID} = lrateb2.*NN.db2{blockID};

                NN.deltaW4E{blockID} = NN.rho*NN.deltaW4E{blockID} + (1-NN.rho)*NN.deltaW4{blockID}.^2;
                NN.deltaW3E{blockID} = NN.rho*NN.deltaW3E{blockID} + (1-NN.rho)*NN.deltaW3{blockID}.^2;
                NN.deltab3E{blockID} = NN.rho*NN.deltab3E{blockID} + (1-NN.rho)*NN.deltab3{blockID}.^2;
                NN.deltaW2E{blockID} = NN.rho*NN.deltaW2E{blockID} + (1-NN.rho)*NN.deltaW2{blockID}.^2;
                NN.deltab2E{blockID} = NN.rho*NN.deltab2E{blockID} + (1-NN.rho)*NN.deltab2{blockID}.^2;
                
                NN.W4{blockID} = NN.W4{blockID}+NN.deltaW4{blockID};    
                NN.W3{blockID} = NN.W3{blockID}+NN.deltaW3{blockID};    
                NN.b3{blockID} = NN.b3{blockID}+NN.deltab3{blockID};
                NN.W2{blockID} = NN.W2{blockID}+NN.deltaW2{blockID};    
                NN.b2{blockID} = NN.b2{blockID}+NN.deltab2{blockID};
            end
            
        elseif(strcmp(setting.alg, 'adam'))
        end
    else
        error('posterior must be either "block" or "diag"');
    end
end
