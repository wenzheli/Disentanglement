% load pretrained model from image to hidden layer h
load ../../data/faceData1K.mat

dataTr = hiddenFace;
clear hiddenFace;

nSamples = size(dataTr,2);
mbSize = 50;
numBatch = floor(nSamples/mbSize);

% model architecture
nBlocks = 1;
D0 = size(dataTr,1); % 784
D1 = 32;            % h1
L = 10;               % L copies Z_{l}|X 

% model variables
X1      = zeros(D0, mbSize);
for blockID = 1:nBlocks
    eta{blockID}    = zeros(D1, mbSize);
    AM{blockID}     = zeros(D1^2, mbSize);
    AT{blockID}     = zeros(D1, D1, mbSize);
    Sigma{blockID}  = zeros(D1, D1, mbSize);
    Z{blockID}      = zeros(D1, mbSize*L);
end
Y       = zeros(D0, mbSize*L);

% hyperparameters in ADA, selected from experiments without prior

adaParams = [
    0.3 1e-8;
    0.3 1e-5;
    0.7 1e-8;
    0.7 1e-5];

    %% ADA learning
nADA = size(adaParams,1);

nwishParams = [10 10;
    50 50;
    500 500];
nWish = size(nwishParams,1);

for pid = 1:nWish
    for hid = 1:nADA
        rho = adaParams(hid,1);    % (1-rho) is weight of current batch
        const = adaParams(hid,2);  % constant used to estimate RMS(.)
        fprintf(2,'AdaDelta parameters: %f, %s\n', rho, num2str(const));
        %% initialize the hyperparameters
        for blockID = 1:nBlocks
            WW{blockID} = eye(D1)/nwishParams(pid,2);
            Wmu{blockID} = zeros(D1,1);
            Wbeta{blockID} = nwishParams(pid,1);
            Wnu{blockID} = nwishParams(pid,2);
        end

        %% initialize the parameters
        for blockID = 1:nBlocks
            W1{blockID} = rand(D0, D1)-0.5;
            b1{blockID} = rand(D1, 1);
            W2{blockID} = rand(D0, D1^2)-0.5;
            b2{blockID} = rand(D1^2,1);
            W3{blockID} = rand(D1, D0)-0.5;
        end
        b3 = rand(D0, 1);

        for blockID = 1:nBlocks
            dW1{blockID} = W1{blockID}*0;
            db1{blockID} = b1{blockID}*0;
            dW2{blockID} = W2{blockID}*0;
            db2{blockID} = b2{blockID}*0;
            dW3{blockID} = W3{blockID}*0;
        end
        db3 = b3*0;

        %% declare & initialize variables used in AdaDelta algorithm
        db3E = b3*0;
        deltab3 = b3*0;
        deltab3E = b3*0;

        for blockID = 1:nBlocks
            dW1E{blockID} = W1{blockID}*0; db1E{blockID} = b1{blockID}*0;
            dW2E{blockID} = W2{blockID}*0; db2E{blockID} = b2{blockID}*0;
            dW3E{blockID} = W3{blockID}*0; 

            deltaW1{blockID} = W1{blockID}*0; deltab1{blockID} = b1{blockID}*0; 
            deltaW2{blockID} = W2{blockID}*0; deltab2{blockID} = b2{blockID}*0;
            deltaW3{blockID} = W3{blockID}*0;

            deltaW1E{blockID} = W1{blockID}*0; deltab1E{blockID} = b1{blockID}*0;
            deltaW2E{blockID} = W2{blockID}*0; deltab2E{blockID} = b2{blockID}*0;
            deltaW3E{blockID} = W3{blockID}*0;
        end

        %% declare other variables
        deltaH      = zeros(D0, mbSize*L);
        for blockID = 1:nBlocks
            deltaZ{blockID}   = zeros(D1, mbSize*L);
            deltaEta{blockID}  = zeros(D1, mbSize);
            deltaAT{blockID}  = zeros(D1, D1, mbSize);
            deltaAM{blockID}  = zeros(D1^2, mbSize);
            deltaBeta{blockID}  = zeros(D1^2, mbSize);
        end

        nEpoch = 20;
        LL = zeros(nEpoch,1);
        hist.E = 0;

        %% ADA loop
        earlyStop = 0;
        for epoch = 1:nEpoch
            index = randperm(nSamples);
            for batchIdx = 1:numBatch
                if(earlyStop==0)
                    firstIdx = (batchIdx-1)*mbSize+1;
                    lastIdx = batchIdx*mbSize;

                    X1 = dataTr(:, index(firstIdx:lastIdx));

                    for blockID = 1:nBlocks
                        eta{blockID} = bsxfun(@plus, W1{blockID}'*X1, b1{blockID});
                        Beta{blockID} = bsxfun(@plus, W2{blockID}'*X1, b2{blockID});
                        AM{blockID} = exp(Beta{blockID});

                        AT{blockID} = reshape(AM{blockID}, [D1, D1, mbSize]);
                        epsilon{blockID} = randn(D1, mbSize*L);
                        for i=1:mbSize
                            Sigma{blockID}(:,:,i) = AT{blockID}(:,:,i)*AT{blockID}(:,:,i)';
                            id1 = (i-1)*L+1;
                            id2 = i*L;
                            Z{blockID}(:, id1:id2) = bsxfun(@plus, AT{blockID}(:,:,i)*epsilon{blockID}(:,id1:id2), eta{blockID}(:,i));
                        end
                    end

                    h = bsxfun(@plus, W3{1}'*Z{1}, b3);
                    for blockID = 2:nBlocks
                        h = h + W3{blockID}'*Z{blockID};
                    end
                    Y = 1./(1+exp(-h));

                    for i=1:mbSize
                        id1 = (i-1)*L+1;
                        id2 = i*L;
                        X2(:,id1:id2) = repmat(X1(:,i),[1,L]);
                    end

                    energy = sum(sum(X2.*log(Y+1e-32) + (1-X2).*log(1-Y+1e-32)));

                    LL(epoch) = LL(epoch) + energy;
    %                 hist.E = [hist.E; LL(epoch)/mbSize/batchIdx];
    %                 if(epoch>=8 && LL(epoch)/mbSize/batchIdx/L <= -205)
    %                     earlyStop =1;
    %                     continue
    %                 end

                    if(isnan(sum(LL(epoch))) || isinf(sum(LL(epoch))))
                        earlyStop =1;
                        continue
                    end

    %                 if(rem(batchIdx,400)==0)
    %                     fprintf('epoch %d, batch %d, energy %f\n', ...
    %                         epoch, batchIdx, LL(epoch)/mbSize/batchIdx/L);
    %                 end

                    %% backpropagation - 1: w.r.t. layers
                    deltaH = X2-Y;
                    % estimate delta w.r.t mu and A
                    for blockID = 1:nBlocks
                        deltaZ{blockID} = W3{blockID}*deltaH;
                        deltaEta{blockID} = deltaEta{blockID}*0;
                        deltaAT{blockID} = deltaAT{blockID}*0;
                        for i=1:mbSize
                            id1 = (i-1)*L+1;
                            id2 = i*L;

                            % gradients from reconstruction Log-Likelihood
                            deltaAT{blockID}(:,:,i) = ...
                                deltaZ{blockID}(:,id1:id2)*transpose(epsilon{blockID}(:,id1:id2))/L;
                            deltaEta{blockID}(:,i) = ...
                                sum(deltaZ{blockID}(:,id1:id2),2)/L;

                            % gradients from minus KL divergence
                            deltaAT{blockID}(:,:,i) = deltaAT{blockID}(:,:,i) ...
                                + transpose(inv(AT{blockID}(:,:,i)+eye(D1)*1e-32)) ...
                                - Wnu{blockID}*WW{blockID}*AT{blockID}(:,:,i);
                            deltaEta{blockID}(:,i) = deltaEta{blockID}(:,i) ...
                                - Wnu{blockID}*WW{blockID}*(eta{blockID}(:,i)-Wmu{blockID});
                        end
                        deltaAM{blockID} = reshape(deltaAT{blockID},[D1^2, mbSize]);
                        deltaBeta{blockID} = deltaAM{blockID}.*AM{blockID};
                    end

                    %% backpropagation - 2: w.r.t. parameters
                    for blockID = 1:nBlocks
                        dW3{blockID} = Z{blockID}*deltaH'/mbSize/L;
                        dW2{blockID} = X1*deltaBeta{blockID}'/mbSize;
                        dW1{blockID} = X1*deltaEta{blockID}'/mbSize;
                    end

                    db3 = sum(deltaH,2)/mbSize/L;
                    for blockID = 1:nBlocks
                        db2{blockID} = sum(deltaBeta{blockID},2)/mbSize;
                        db1{blockID} = sum(deltaEta{blockID},2)/mbSize;
                    end

                    %% ADA update of parameters and statistics
                    db3E = rho*db3E + (1-rho)*db3.^2;
                    lrateb3 = sqrt(deltab3E+const)./sqrt(db3E+const);
                    deltab3 = lrateb3.*db3;
                    deltab3E= rho*deltab3E + (1-rho)*deltab3.^2;

                    for blockID = 1:nBlocks  
                        dW3E{blockID} = rho*dW3E{blockID} + (1-rho)*dW3{blockID}.^2;   
                        dW2E{blockID} = rho*dW2E{blockID} + (1-rho)*dW2{blockID}.^2;   
                        db2E{blockID} = rho*db2E{blockID} + (1-rho)*db2{blockID}.^2;
                        dW1E{blockID} = rho*dW1E{blockID} + (1-rho)*dW1{blockID}.^2;   
                        db1E{blockID} = rho*db1E{blockID} + (1-rho)*db1{blockID}.^2;

                        lrateW3{blockID} = sqrt(deltaW3E{blockID}+const)./sqrt(dW3E{blockID}+const);
                        lrateW2{blockID} = sqrt(deltaW2E{blockID}+const)./sqrt(dW2E{blockID}+const);
                        lrateW1{blockID} = sqrt(deltaW1E{blockID}+const)./sqrt(dW1E{blockID}+const);

                        lrateb2{blockID} = sqrt(deltab2E{blockID}+const)./sqrt(db2E{blockID}+const);
                        lrateb1{blockID} = sqrt(deltab1E{blockID}+const)./sqrt(db1E{blockID}+const);

                        deltaW3{blockID} = lrateW3{blockID}.*dW3{blockID};
                        deltaW2{blockID} = lrateW2{blockID}.*dW2{blockID}; 
                        deltab2{blockID} = lrateb2{blockID}.*db2{blockID};
                        deltaW1{blockID} = lrateW1{blockID}.*dW1{blockID};
                        deltab1{blockID} = lrateb1{blockID}.*db1{blockID};

                        deltaW3E{blockID} = rho*deltaW3E{blockID} + (1-rho)*deltaW3{blockID}.^2;
                        deltaW2E{blockID} = rho*deltaW2E{blockID} + (1-rho)*deltaW2{blockID}.^2;
                        deltaW1E{blockID} = rho*deltaW1E{blockID} + (1-rho)*deltaW1{blockID}.^2;

                        deltab2E{blockID} = rho*deltab2E{blockID} + (1-rho)*deltab2{blockID}.^2;
                        deltab1E{blockID} = rho*deltab1E{blockID} + (1-rho)*deltab1{blockID}.^2;
                    end

                    b3 = b3+deltab3;
                    for blockID = 1:nBlocks
                        W3{blockID} = W3{blockID}+deltaW3{blockID};    
                        W2{blockID} = W2{blockID}+deltaW2{blockID};    b2{blockID} = b2{blockID}+deltab2{blockID};
                        W1{blockID} = W1{blockID}+deltaW1{blockID};    b1{blockID} = b1{blockID}+deltab1{blockID};
                    end

                    %% backpropagation - 3: update the hyperparameters
                    for blockID = 1:nBlocks
                        S1 = sum(Sigma{blockID},3);
                        Sy = mean(eta{blockID},2)-Wmu{blockID};
                        S2 = Sy*Sy';
                        Wmu{blockID} = Wbeta{blockID}/(Wbeta{blockID}+mbSize)*Wmu{blockID} ...
                            + mbSize/(Wbeta{blockID}+mbSize)*mean(eta{blockID},2); % convex comb
                        WW{blockID} = inv(inv(WW{blockID}) + S1 + Wbeta{blockID}*mbSize/(Wbeta{blockID}+mbSize)*S2);
                        Wbeta{blockID} = Wbeta{blockID}+mbSize;
                        Wnu{blockID} = Wnu{blockID} + mbSize;
                    end

                end
            end
            fprintf('epoch %d, log-likelihood is %f\n', epoch, LL(epoch)/nSamples/L);
        end
        model{hid}.W3 = W3; model{hid}.b3 = b3;
        model{hid}.W2 = W2; model{hid}.b2 = b2;
        model{hid}.W1 = W1; model{hid}.b1 = b1;
        model{hid}.LL = LL/nSamples/L;
        model{hid}.WW = WW;
        model{hid}.Wmu = Wmu;
        model{hid}.Wbeta = Wbeta;
        model{hid}.Wnu = Wnu;
        
    %     model{hid}.energy = hist.E(2:end);

        save(['ptNW' num2str(pid) 'PriorBlock' num2str(nBlocks) 'Dim' num2str(D1) '.mat'],'model');
    end
end
