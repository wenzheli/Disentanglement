% a demo code of the "exponential full covariance matrix" 
%   together with fixed Gaussian prior distributions

load ../../../week3/AEVB/data/batchtraindata.mat;
dataTr = batchdata';
clear batchdata;

nBlocks = 1;
D0 = size(dataTr,1); % 784
D1 = 200;            % h1
D2 = 2;              % Z
D3 = 200;            % h2
L = 10;               % L copies Z_{l}|X 
mbSize = 50;

X1      = zeros(D0, mbSize);
h1      = zeros(D1, mbSize);
for blockID = 1:nBlocks
    eta{blockID}    = zeros(D2, mbSize);
    Beta{blockID}   = zeros(D2^2, mbSize);
    AM{blockID}     = zeros(D2^2, mbSize);
    AT{blockID}     = zeros(D2, D2, mbSize);
    Sigma{blockID}  = zeros(D2, D2, mbSize);
    Z{blockID}      = zeros(D2, mbSize*L);
    Z{blockID}      = zeros(D2, mbSize*L);
end
h2      = zeros(D3, mbSize*L);
X2      = zeros(D0, mbSize*L);
Y       = zeros(D0, mbSize*L);

% hyperparameters in SGD, selected from experiments without prior
nwishParams = [50 50;
    500 500;
    5000 5000];

sgdParams = [0.1 1e-8;
    0.3 1e-8;
    0.5 1e-8];

%% setup prior hyperparameters
for blockID = 1:nBlocks
    PLambda{blockID} = eye(D2);
    Pmu{blockID} = zeros(D2,1);
end
    
%% SGD learning
nNWish = size(nwishParams,1);
nSGD = size(sgdParams,1);
    for hid = 1:nSGD
        rho = sgdParams(hid,1);    % (1-rho) is weight of current batch
        const = sgdParams(hid,2);  % constant used to estimate RMS(.)
        
        %% initialize the parameters
        W1 = rand(D0, D1)-0.5;      b1 = rand(D1, 1);
        for blockID = 1:nBlocks
            W2{blockID} = rand(D1, D2)-0.5;   
            b2{blockID} = rand(D2, 1);
            W3{blockID} = rand(D1, D2^2)-0.5;    
            b3{blockID} = rand(D2^2,1);
            W4{blockID} = rand(D2, D3)-0.5;   
        end
        b4 = rand(D3, 1);
        W5 = rand(D3, D0)-0.5;      b5 = rand(D0, 1);

        dW5 = W5*0; db5 = b5*0;
        db4 = b4*0;
        for blockID = 1:nBlocks
            dW4{blockID} = W4{blockID}*0; 
            dW3{blockID} = W3{blockID}*0; 
            db3{blockID} = b3{blockID}*0;
            dW2{blockID} = W2{blockID}*0;
            db2{blockID} = b2{blockID}*0;
        end
        dW1 = W1*0; db1 = b1*0;

        %% declare & initialized variables used in adadelta algorithm
        dW5E = W5*0; db5E = b5*0;
        dW1E = W1*0; db1E = b1*0;

        deltaW5 = W5*0; deltab5 = b5*0;
        deltaW1 = W1*0; deltab1 = b1*0;

        deltaW5E = W5*0; deltab5E = b5*0;
        deltaW1E = W1*0; deltab1E = b1*0;

        db4E = b4*0;
        deltab4 = b4*0;
        deltab4E = b4*0;

        for blockID = 1:nBlocks
            dW4E{blockID} = W4{blockID}*0;
            dW3E{blockID} = W3{blockID}*0; db3E{blockID} = b3{blockID}*0;
            dW2E{blockID} = W2{blockID}*0; db2E{blockID} = b2{blockID}*0;

            deltaW4{blockID} = W4{blockID}*0;
            deltaW3{blockID} = W3{blockID}*0; deltab3{blockID} = b3{blockID}*0;
            deltaW2{blockID} = W2{blockID}*0; deltab2{blockID} = b2{blockID}*0;

            deltaW4E{blockID} = W4{blockID}*0;
            deltaW3E{blockID} = W3{blockID}*0; deltab3E{blockID} = b3{blockID}*0;
            deltaW2E{blockID} = W2{blockID}*0; deltab2E{blockID} = b2{blockID}*0;
        end

        %% declare other variables
        delta3      = zeros(D0, mbSize*L);
        delta2      = zeros(D3, mbSize*L);
        for blockID = 1:nBlocks
            delta1{blockID}   = zeros(D2, mbSize*L);
            deltaEta{blockID}  = zeros(D2, mbSize);
            deltaAT{blockID}  = zeros(D2, D2, mbSize);
            deltaAM{blockID}  = zeros(D2^2, mbSize);
            deltaBeta{blockID}  = zeros(D2^2, mbSize);
        end
        delta0	= zeros(D1, mbSize);

        nSamples = size(dataTr,2);
        numBatch = floor(nSamples/mbSize);

        nEpoch = 15;
        LL = zeros(nEpoch,4);
        energy = zeros(1,4);
        hist.E = zeros(1,4);

        %% SGD loop
        for epoch = 1:nEpoch
            index = randperm(nSamples);
            for batchIdx = 1:numBatch
                firstIdx = (batchIdx-1)*mbSize+1;
                lastIdx = batchIdx*mbSize;

                X1 = dataTr(:, index(firstIdx:lastIdx));
                h1 = tanh(bsxfun(@plus, W1'*X1, b1));

                for blockID = 1:nBlocks
                    eta{blockID} = bsxfun(@plus, W2{blockID}'*h1, b2{blockID});
                    Beta{blockID} = bsxfun(@plus, W3{blockID}'*h1, b3{blockID});
                    AM{blockID} = exp(Beta{blockID});
                    AT{blockID} = reshape(AM{blockID}, [D2, D2, mbSize]);
                    epsilon{blockID} = randn(D2, mbSize*L);
                    for i=1:mbSize
                        id1 = (i-1)*L+1;
                        id2 = i*L;
                        Z{blockID}(:, id1:id2) = bsxfun(@plus, AT{blockID}(:,:,i)*epsilon{blockID}(:,id1:id2), eta{blockID}(:,i));
                    end
                end

                h2 = bsxfun(@plus, W4{1}'*Z{1}, b4);
                for blockID = 2:nBlocks
                    h2 = h2 + W4{blockID}'*Z{blockID};
                end
                h2 = tanh(h2);

                Y = 1./(1+exp(-bsxfun(@plus, W5'*h2, b5)));

                for i=1:mbSize
                    id1 = (i-1)*L+1;
                    id2 = i*L;
                    X2(:,id1:id2) = repmat(X1(:,i),[1,L]);
                end

                % calculate the objective function on this mb data
                % inlcuding reconstruction loss part 
                %   and KL divergence parts
                energy = energy*0;
                energy(1) = sum(sum(X2.*log(Y+1e-32) + (1-X2).*log(1-Y+1e-32)));
                for blockID = 1:nBlocks
                    for i=1:mbSize
                        Sigma{blockID}(:,:,i) = AT{blockID}(:,:,i)*AT{blockID}(:,:,i)';
                        energy(2) = energy(2) + 0.5*log(det(Sigma{blockID}(:,:,i))+1e-16);
                        energy(3) = energy(3) - 0.5*trace(PLambda{blockID}*Sigma{blockID}(:,:,i));
                        energy(4) = energy(4) - ...
                            0.5*(eta{blockID}(:,i) - Pmu{blockID})'*PLambda{blockID}*(eta{blockID}(:,i)-Pmu{blockID});
                    end
                end
                
                LL(epoch,:) = LL(epoch,:) + energy;
                hist.E = [hist.E; LL(epoch,:)/mbSize/batchIdx];
                if(rem(batchIdx,400)==0)
                    fprintf('epoch %d, batch %d, energy %f, %f, %f, %f\n', ...
                        epoch, batchIdx, LL(epoch, 1)/mbSize/batchIdx/L, ...
                        LL(epoch, 2)/mbSize/batchIdx, LL(epoch, 3)/mbSize/batchIdx, ...
                        LL(epoch, 4)/mbSize/batchIdx);
                end
                
                %% backpropagation - 1: w.r.t. layers
                delta3 = X2-Y;
                delta2 = (W5*delta3).*(1-h2.^2);
                % estimate delta w.r.t mu and A
                for blockID = 1:nBlocks
                    delta1{blockID} = W4{blockID}*delta2;
                    deltaEta{blockID} = deltaEta{blockID}*0;
                    deltaAT{blockID} = deltaAT{blockID}*0;
                    for i=1:mbSize
                        id1 = (i-1)*L+1;
                        id2 = i*L;

                        % gradients from reconstruction Log-Likelihood
                        deltaAT{blockID}(:,:,i) = ...
                            delta1{blockID}(:,id1:id2)*transpose(epsilon{blockID}(:,id1:id2))/L;
                        deltaEta{blockID}(:,i) = ...
                            sum(delta1{blockID}(:,id1:id2),2)/L;
                        
                        % gradients from minus KL divergence
                        deltaAT{blockID}(:,:,i) = deltaAT{blockID}(:,:,i) + ...
                            transpose(inv(AT{blockID}(:,:,i)+eye(D2)*1e-32)) ... 
                            -PLambda{blockID}*AT{blockID}(:,:,i);
                        deltaEta{blockID}(:,i) = deltaEta{blockID}(:,i) - ...
                            PLambda{blockID}*(eta{blockID}(:,i)-Pmu{blockID});
                    end
                    deltaAM{blockID} = reshape(deltaAT{blockID},[D2^2, mbSize]);
                    deltaBeta{blockID} = deltaAM{blockID}.*AM{blockID};
                end

                delta0 = W2{1}*deltaEta{1} + W3{1}*deltaBeta{1};
                for blockID = 2:nBlocks
                    delta0 = delta0 + W2{blockID}*deltaEta{blockID} + W3{blockID}*deltaBeta{blockID};
                end
                delta0 = delta0.*(1-h1.^2);

                %% backpropagation - 2: w.r.t. parameters
                dW5 = h2*delta3'/mbSize/L;
                for blockID = 1:nBlocks
                    dW4{blockID} = Z{blockID}*delta2'/mbSize/L;
                    dW3{blockID} = h1*deltaBeta{blockID}'/mbSize;
                    dW2{blockID} = h1*deltaEta{blockID}'/mbSize;
                end
                dW1 = X1*delta0'/mbSize;

                db5 = sum(delta3,2)/mbSize/L;
                db4 = sum(delta2,2)/mbSize/L;
                for blockID = 1:nBlocks
                    db3{blockID} = sum(deltaBeta{blockID},2)/mbSize;
                    db2{blockID} = sum(deltaEta{blockID},2)/mbSize;
                end
                db1 = sum(delta0,2)/mbSize;


                %% ADA update of parameters and statistics
                dW5E = rho*dW5E + (1-rho)*dW5.^2;   
                db5E = rho*db5E + (1-rho)*db5.^2;
                dW1E = rho*dW1E + (1-rho)*dW1.^2;   
                db1E = rho*db1E + (1-rho)*db1.^2;

                lrateW5 = sqrt(deltaW5E+const)./sqrt(dW5E+const);
                lrateW1 = sqrt(deltaW1E+const)./sqrt(dW1E+const);
                lrateb5 = sqrt(deltab5E+const)./sqrt(db5E+const);
                lrateb1 = sqrt(deltab1E+const)./sqrt(db1E+const);

                deltaW5 = lrateW5.*dW5; 
                deltab5 = lrateb5.*db5;
                deltaW1 = lrateW1.*dW1; 
                deltab1 = lrateb1.*db1;

                deltaW5E = rho*deltaW5E + (1-rho)*deltaW5.^2;
                deltaW1E = rho*deltaW1E + (1-rho)*deltaW1.^2;
                deltab5E = rho*deltab5E + (1-rho)*deltab5.^2;
                deltab1E = rho*deltab1E + (1-rho)*deltab1.^2;

                db4E = rho*db4E + (1-rho)*db4.^2;
                lrateb4 = sqrt(deltab4E+const)./sqrt(db4E+const);
                deltab4 = lrateb4.*db4;
                deltab4E= rho*deltab4E + (1-rho)*deltab4.^2;
                for blockID = 1:nBlocks
                    dW4E{blockID} = rho*dW4E{blockID} + (1-rho)*dW4{blockID}.^2;   
                    dW3E{blockID} = rho*dW3E{blockID} + (1-rho)*dW3{blockID}.^2;   
                    db3E{blockID} = rho*db3E{blockID} + (1-rho)*db3{blockID}.^2;
                    dW2E{blockID} = rho*dW2E{blockID} + (1-rho)*dW2{blockID}.^2;   
                    db2E{blockID} = rho*db2E{blockID} + (1-rho)*db2{blockID}.^2;

                    lrateW4{blockID} = sqrt(deltaW4E{blockID}+const)./sqrt(dW4E{blockID}+const);
                    lrateW3{blockID} = sqrt(deltaW3E{blockID}+const)./sqrt(dW3E{blockID}+const);
                    lrateW2{blockID} = sqrt(deltaW2E{blockID}+const)./sqrt(dW2E{blockID}+const);

                    lrateb3{blockID} = sqrt(deltab3E{blockID}+const)./sqrt(db3E{blockID}+const);
                    lrateb2{blockID} = sqrt(deltab2E{blockID}+const)./sqrt(db2E{blockID}+const);

                    deltaW4{blockID} = lrateW4{blockID}.*dW4{blockID}; 
                    deltaW3{blockID} = lrateW3{blockID}.*dW3{blockID};
                    deltab3{blockID} = lrateb3{blockID}.*db3{blockID};
                    deltaW2{blockID} = lrateW2{blockID}.*dW2{blockID}; 
                    deltab2{blockID} = lrateb2{blockID}.*db2{blockID};

                    deltaW4E{blockID} = rho*deltaW4E{blockID} + (1-rho)*deltaW4{blockID}.^2;
                    deltaW3E{blockID} = rho*deltaW3E{blockID} + (1-rho)*deltaW3{blockID}.^2;
                    deltaW2E{blockID} = rho*deltaW2E{blockID} + (1-rho)*deltaW2{blockID}.^2;

                    deltab3E{blockID} = rho*deltab3E{blockID} + (1-rho)*deltab3{blockID}.^2;
                    deltab2E{blockID} = rho*deltab2E{blockID} + (1-rho)*deltab2{blockID}.^2;
                end

                W5 = W5+deltaW5;    b5 = b5+deltab5;
                b4 = b4+deltab4;
                for blockID = 1:nBlocks
                    W4{blockID} = W4{blockID}+deltaW4{blockID};    
                    W3{blockID} = W3{blockID}+deltaW3{blockID};    b3{blockID} = b3{blockID}+deltab3{blockID};
                    W2{blockID} = W2{blockID}+deltaW2{blockID};    b2{blockID} = b2{blockID}+deltab2{blockID};
                end
                W1 = W1+deltaW1;    b1 = b1+deltab1;

            end
            fprintf('epoch %d, log-likelihood is %f, %f, %f, %f, %f\n', epoch, ...
                LL(epoch,1)/nSamples/L, LL(epoch,2)/nSamples, LL(epoch,3)/nSamples, ...
                LL(epoch,4)/nSamples, (LL(epoch,1)/L+sum(LL(epoch,2:end)))/mbSize/batchIdx);

        end

        model{hid}.W5 = W5; model{hid}.b5 = b5;
        model{hid}.W4 = W4; model{hid}.b4 = b4;
        model{hid}.W3 = W3; model{hid}.b3 = b3;
        model{hid}.W2 = W2; model{hid}.b2 = b2;
        model{hid}.W1 = W1; model{hid}.b1 = b1;
        model{hid}.LL(:,1) = LL(:,1)/nSamples/L;
        model{hid}.LL(:,2:4) = LL(:,2:4)/nSamples;
        model{hid}.energy = hist.E(2:end,:);
    end
    save(['model5AdaFix.mat'],'model');
