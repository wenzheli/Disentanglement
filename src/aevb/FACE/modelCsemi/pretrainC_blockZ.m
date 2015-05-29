function [NN] = pretrainC_blockZ(NN, NNsetting, dataTr, labelTr)
% pretrain the H1-->Z-->H2 layer, 
%   i.e. the middle dim reduction layer in modelC
%   Block-NW prior is used 

    NW = defaultNWinit(NN);
    mbSize = NNsetting.mbSize;
    nSamples = size(dataTr,2);
    numBatch = floor(nSamples/mbSize);

    dataH = zeros(NN.D1, nSamples);
    for firstIdx = 1:1000:nSamples
        lastIdx = min(firstIdx+999, nSamples);
        dataH(:,firstIdx:lastIdx) = 1./(1+exp(-bsxfun(@plus, NN.W1'*dataTr(:,firstIdx:lastIdx), NN.b1)));
    end
    
    quickStop=0;
    NN.ptZLossRecon = zeros(NNsetting.nPtEpoch,1);
    NN.ptZLossPred = zeros(NNsetting.nPtEpoch,NN.nClasses);
    
    for epoch = 1:NNsetting.nPtEpoch
        if(quickStop==1)
            break;
        end

        index = randperm(nSamples);
        for batchIdx = 1:numBatch
            firstIdx = (batchIdx-1)*mbSize+1;
            lastIdx = batchIdx*mbSize;
            if(quickStop==1)
                break;
            end
            NN.h1 = dataH(:, index(firstIdx:lastIdx));
            
            for m=1:NN.nClasses
                yTr{m} = labelTr{m}(:, index(firstIdx:lastIdx));
            end
            NN = ffBlockZ(NN, NNsetting, yTr, NN.supervised(index(firstIdx:lastIdx)));
            NN.ptZLossRecon(epoch)  = NN.ptZLossRecon(epoch)+NN.lossRecon;
            NN.ptZLossPred(epoch,:)   = NN.ptZLossPred(epoch,:)+NN.lossPred;
            if(rem(batchIdx,120)==1)
                fprintf(2,'epoch %d, minibatch %d, recon loss: %f,[ %s ]\n', ...
                    epoch, batchIdx, ...
                    NN.ptZLossRecon(epoch)/batchIdx/mbSize/NNsetting.L,...
                    num2str(NN.ptZLossPred(epoch,:)/batchIdx/mbSize/NNsetting.L/NN.ratioLabel));
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

function NN = ffBlockZ(NN, setting, Y1, supervised)

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
    
    for classID=1:NN.nClasses
        NN.predY{classID} = bsxfun(@plus, NN.Wc{classID}'*NN.Z{NN.mapBlock(classID)}, NN.bc{classID});
        NN.predY{classID} = exp(bsxfun(@minus, NN.predY{classID}, max(NN.predY{classID})));
        NN.predY{classID} = bsxfun(@rdivide, NN.predY{classID}, sum(NN.predY{classID}));
    end

    NN.h2 = bsxfun(@plus, NN.W4{1}'*NN.Z{1}, NN.b4);
    for blockID = 2:NN.nBlocks
        NN.h2 = NN.h2 + NN.W4{blockID}'*NN.Z{blockID};
    end
    NN.h2 = 1./(1+exp(-NN.h2));
    
    idx = (0:mbSize-1)*L;
    for i=1:L
        NN.delta2(:,idx+i) = NN.h1;
        for classID=1:NN.nClasses
            NN.deltaC{classID}(:,idx+i) = Y1{classID};
        end
    end

    
    NN.lossRecon = -sum(sum(NN.delta2.*log(NN.h2+1e-32) + (1-NN.delta2).*log(1-NN.h2+1e-32)));
    NN.delta2 = NN.delta2-NN.h2;
    
    %     NN.lossPred = -sum(sum(NN.deltaC.*log(NN.predY+1e-32) + (1-NN.deltaC).*log(1-NN.predY)));
    supervised = transpose(reshape(repmat(supervised,[L,1]), mbSize*L,1));
    for classID=1:NN.nClasses
        [~, l1{classID}] = max(NN.deltaC{classID}); % true classes
        [~, l2{classID}] = max(NN.predY{classID});  % pred classes
        NN.lossPred(classID) = sum((l1{classID}~=l2{classID}).*supervised);
        NN.deltaC{classID} = NN.cWeight(classID)*bsxfun(@times, (NN.deltaC{classID} - NN.predY{classID}), supervised);
    end
end

function NN = backBlockZ(NN, setting, NW)
    L = setting.L;
    mbSize = setting.mbSize;
    
    % gradient w.r.t layer Z, Reconstruction part
    for blockID = 1:NN.nBlocks
        NN.delta1{blockID} = NN.W4{blockID}*NN.delta2;
    end
    % gradient w.r.t classifier, Classification part
    % temporarily, only consider one group of label
    for classID = 1:NN.nClasses
        NN.delta1{NN.mapBlock(classID)} = NN.delta1{classID} + NN.Wc{classID}*NN.deltaC{classID};
    end

    for blockID = 1:NN.nBlocks
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

    
    for classID = 1:NN.nClasses
        NN.dWc{classID} = NN.Z{NN.mapBlock(classID)}*NN.deltaC{classID}'/mbSize/L;
        NN.dbc{classID} = sum(NN.deltaC{classID},2)/mbSize/L;
    end

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
            for classID = NN.nClasses
                NN.Wc{classID} = NN.Wc{classID} + NN.lrate.*NN.dWc; NN.bc = NN.bc + NN.lrate.*NN.dbc;
            end
        elseif(strcmp(setting.alg, 'adadelta'))
            %% ADA update of parameters and statistics


            NN.dW4E = NN.rho*NN.dW4E + (1-NN.rho)*NN.dW4.^2;   
            NN.dW3E = NN.rho*NN.dW3E + (1-NN.rho)*NN.dW3.^2;   
            NN.dW2E = NN.rho*NN.dW2E + (1-NN.rho)*NN.dW2.^2;   
            
            NN.db4E = NN.rho*NN.db4E + (1-NN.rho)*NN.db4.^2;
            NN.db3E = NN.rho*NN.db3E + (1-NN.rho)*NN.db3.^2;
            NN.db2E = NN.rho*NN.db2E + (1-NN.rho)*NN.db2.^2;

            lrateW4 = sqrt(NN.deltaW4E+NN.const)./sqrt(NN.dW4E+NN.const);
            lrateW3 = sqrt(NN.deltaW3E+NN.const)./sqrt(NN.dW3E+NN.const);
            lrateW2 = sqrt(NN.deltaW2E+NN.const)./sqrt(NN.dW2E+NN.const);
            
            lrateb4 = sqrt(NN.deltab4E+NN.const)./sqrt(NN.db4E+NN.const);
            lrateb3 = sqrt(NN.deltab3E+NN.const)./sqrt(NN.db3E+NN.const);
            lrateb2 = sqrt(NN.deltab2E+NN.const)./sqrt(NN.db2E+NN.const);

            NN.deltaW4 = lrateW4.*NN.dW4; 
            NN.deltaW3 = lrateW3.*NN.dW3;
            NN.deltaW2 = lrateW2.*NN.dW2; 
            
            
            NN.deltab4 = lrateb4.*NN.db4;
            NN.deltab3 = lrateb3.*NN.db3;
            NN.deltab2 = lrateb2.*NN.db2;

            NN.deltaW4E = NN.rho*NN.deltaW4E + (1-NN.rho)*NN.deltaW4.^2;
            NN.deltaW3E = NN.rho*NN.deltaW3E + (1-NN.rho)*NN.deltaW3.^2;
            NN.deltaW2E = NN.rho*NN.deltaW2E + (1-NN.rho)*NN.deltaW2.^2;

            NN.deltab4E = NN.rho*NN.deltab4E + (1-NN.rho)*NN.deltab4.^2;
            NN.deltab3E = NN.rho*NN.deltab3E + (1-NN.rho)*NN.deltab3.^2;
            NN.deltab2E = NN.rho*NN.deltab2E + (1-NN.rho)*NN.deltab2.^2;

            NN.W4 = NN.W4+NN.deltaW4;    NN.b4 = NN.b4+NN.deltab4;
            NN.W3 = NN.W3+NN.deltaW3;    NN.b3 = NN.b3+NN.deltab3;
            NN.W2 = NN.W2+NN.deltaW2;    NN.b2 = NN.b2+NN.deltab2;
            for classID = 1:NN.nClasses
                
                NN.dWcE{classID} = NN.rho*NN.dWcE{classID} + (1-NN.rho)*NN.dWc{classID}.^2;   
                NN.dbcE{classID} = NN.rho*NN.dbcE{classID} + (1-NN.rho)*NN.dbc{classID}.^2;

                NN.lrateWc{classID} = sqrt(NN.deltaWcE{classID}+NN.const)./sqrt(NN.dWcE{classID}+NN.const);
                NN.lratebc{classID} = sqrt(NN.deltabcE{classID}+NN.const)./sqrt(NN.dbcE{classID}+NN.const);

                NN.deltaWc{classID} = NN.lrateWc{classID}.*NN.dWc{classID}; 
                NN.deltabc{classID} = NN.lratebc{classID}.*NN.dbc{classID};
                
                NN.deltaWcE{classID} = NN.rho*NN.deltaWcE{classID} + (1-NN.rho)*NN.deltaWc{classID}.^2;
                NN.deltabcE{classID} = NN.rho*NN.deltabcE{classID} + (1-NN.rho)*NN.deltabc{classID}.^2;

                NN.Wc{classID} = NN.Wc{classID} + NN.deltaWc{classID};    
                NN.bc{classID} = NN.bc{classID} + NN.deltabc{classID};
            end
            
        elseif(strcmp(setting.alg, 'adam'))
        end
    elseif(strcmp(NN.shape,'block'))
        if(strcmp(setting.alg, 'sgd'))
            % SGD update parameters
            NN.b4 = NN.b4 + NN.lrate.*NN.db4;
            for blockID = 1:NN.nBlocks
                NN.W4{blockID} = NN.W4{blockID} + NN.lrate*NN.dW4{blockID}; 
                NN.W3{blockID} = NN.W3{blockID} + NN.lrate*NN.dW3{blockID}; 
                NN.b3{blockID} = NN.b3{blockID} + NN.lrate*NN.db3{blockID};
                NN.W2{blockID} = NN.W2{blockID} + NN.lrate*NN.dW2{blockID}; 
                NN.b2{blockID} = NN.b2{blockID} + NN.lrate*NN.db2{blockID};
            end
            for classID = 1:NN.nClasses
                NN.Wc{classID} = NN.Wc{classID} + NN.lrate*NN.dWc{classID}; 
                NN.bc{classID} = NN.bc{classID} + NN.lrate*NN.dbc{classID};
            end
            
        elseif(strcmp(setting.alg, 'adadelta'))
            %% ADA update of parameters and statistics
            NN.db4E = NN.rho*NN.db4E + (1-NN.rho)*NN.db4.^2;
            lrateb4 = sqrt(NN.deltab4E+NN.const)./sqrt(NN.db4E+NN.const);
            NN.deltab4 = lrateb4.*NN.db4;
            NN.deltab4E= NN.rho*NN.deltab4E + (1-NN.rho)*NN.deltab4.^2;
            NN.b4 = NN.b4+NN.deltab4;
            
            for classID = 1:NN.nClasses
                NN.dWcE{classID} = NN.rho*NN.dWcE{classID} + (1-NN.rho)*NN.dWc{classID}.^2;
                NN.lrateWc{classID} = sqrt(NN.deltaWcE{classID}+NN.const)./sqrt(NN.dWcE{classID}+NN.const);
                NN.deltaWc{classID} = NN.lrateWc{classID}.*NN.dWc{classID};
                NN.deltaWcE{classID}= NN.rho*NN.deltaWcE{classID} + (1-NN.rho)*NN.deltaWc{classID}.^2;
                NN.Wc{classID} = NN.Wc{classID}+NN.deltaWc{classID};

                NN.dbcE{classID} = NN.rho*NN.dbcE{classID} + (1-NN.rho)*NN.dbc{classID}.^2;
                NN.lratebc{classID} = sqrt(NN.deltabcE{classID}+NN.const)./sqrt(NN.dbcE{classID}+NN.const);
                NN.deltabc{classID} = NN.lratebc{classID}.*NN.dbc{classID};
                NN.deltabcE{classID} = NN.rho*NN.deltabcE{classID} + (1-NN.rho)*NN.deltabc{classID}.^2;
                NN.bc{classID} = NN.bc{classID} + NN.deltabc{classID};
            end

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
