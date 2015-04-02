% AEVB implementation on MNIST data, with diagonal Gaussian posterior assumption
%
%   the posterior covariance matrix of hidden layer q(Z|x) is multiple-block diagonal matrix
% the struction of the model
% q(z|x): dim-784 Xs --> dim-200 hu --> dim-2 Gaussian Z
% p(x|z): dim-2 Gaussian Z --> dim-200 hu --> dim-784 Xs

load data/batchtraindata.mat;
dataTr = batchdata';
clear batchdata;

nBlocks = 2;
D0 = size(dataTr,1);            % 784
D1 = 200;                       % size of h1
for blockID = 1:nBlocks
    D2{blockID} = 2;            % size of Z subsets
end
D3 = 200;                       % size of h2
L = 10;                         % L copies of samples ~ Z_{l}|X 
mbSize = 50;


X1      = zeros(D0, mbSize);
h1      = zeros(D1, mbSize);
% split into multiple blocks
for blockID = 1:nBlocks
    mu{blockID}     = zeros(D2{blockID}, mbSize);
    AM{blockID}     = zeros(D2{blockID}^2, mbSize);          % matrix form
    AT{blockID}     = zeros(D2{blockID}, D2{blockID}, mbSize);        % tensor form
    Z{blockID}      = zeros(D2{blockID}, mbSize*L);
    Z{blockID}      = zeros(D2, mbSize*L);
end
h2      = zeros(D3, mbSize*L);
X2      = zeros(D0, mbSize*L);
Y       = zeros(D0, mbSize*L);


params = [1/2500 0.9;
    1/1000 0.9];
% experiments shows that 0.9 momentum is better than 0.5 or 0 momentum
    
for hid = 1:size(params,1);
    lrate = params(hid,1);
    momentum = params(hid,2);

    % initialize the parameters
    W1 = rand(D0, D1)-0.5;      b1 = rand(D1, 1);
    for blockID = 1:nBlocks
        W2{blockID} = rand(D1, D2{blockID})-0.5;   b2{blockID} = rand(D2{blockID}, 1);
        W3{blockID} = zeros(D1, D2{blockID}^2);    b3{blockID} = zeros(D2{blockID}^2,1);
        W4{blockID} = rand(D2{blockID}, D3)-0.5;   b4{blockID} = rand(D3, 1);
    end
    W5 = rand(D3, D0)-0.5;      b5 = rand(D0, 1);
    W6 = rand(D3, D0)-0.5;      b6 = rand(D0, 1);

    dW6 = W6*0; db6 = b6*0;
    dW5 = W5*0; db5 = b5*0;
    for blockID = 1:nBlocks
        dW4{blockID} = W4{blockID}*0; db4{blockID} = b4{blockID}*0;
        dW3{blockID} = W3{blockID}*0; db3{blockID} = b3{blockID}*0;
        dW2{blockID} = W2{blockID}*0; db2{blockID} = b2{blockID}*0;
    end
    dW1 = W1*0; db1 = b1*0;

    nSamples = size(dataTr,2);
    numBatch = floor(nSamples/mbSize);

    delta3      = zeros(D0, mbSize*L);
    delta2      = zeros(D3, mbSize*L);
    for blockID = 1:nBlocks
        delta1{blockID}   = zeros(D2{blockID}, mbSize*L);
        deltaMu{blockID}  = zeros(D2{blockID}, mbSize);
        deltaAT{blockID}  = zeros(D2{blockID}, D2{blockID}, mbSize);
        deltaAM{blockID}  = zeros(D2{blockID}^2, mbSize);
    end
    delta0      = zeros(D1, mbSize);
    
    nEpoch = 30;
    LL = zeros(nEpoch,1);
    for epoch = 1:nEpoch
        index = randperm(nSamples);
        for batchIdx = 1:numBatch
            firstIdx = (batchIdx-1)*mbSize+1;
            lastIdx = batchIdx*mbSize;
            
            X1 = dataTr(:, index(firstIdx:lastIdx));
            
            %% forward propagation 
            % step 1. X (D*mbSize) --> h1 (D1 * mbSize)
            h1 = tanh(bsxfun(@plus, W1'*X1, b1));
            
            % step 2. h1 (D1 * mbSize) --> mu (D2 * mbSize), A (D2^2 * mbSize)
            for blockID = 1:nBlocks
                mu{blockID} = bsxfun(@plus, W2{blockID}'*h1, b2{blockID});
                AM{blockID} = bsxfun(@plus, W3{blockID}'*h1, b3{blockID});
                AT{blockID} = reshape(AM{blockID}, [D2{blockID}, D2{blockID}, mbSize]);

                % step 3. mu (D2 * mbSize), A (D2, D2 * mbSize) --> Z (D2 * [mbSize*L])
                %Z = mu + A*epsilon;  % need more consideration here 
                epsilon{blockID} = randn(D2{blockID}, mbSize*L);
                % sampling:
                for i=1:mbSize
                    id1 = (i-1)*L+1;
                    id2 = i*L;
                    Z{blockID}(:, id1:id2) = bsxfun(@plus, AT{blockID}(:,:,i)*epsilon{blockID}(:,id1:id2), mu{blockID}(:,i));
                end
            end
            
            % step 4. Z (D2 *[mbSize*L]) --> h2(D3 * [mbSize*L])
            h2 = bsxfun(@plus, W4{1}'*Z{1}, b4{1});
            for blockID = 2:nBlocks
                h2 = h2 + bsxfun(@plus, W4{blockID}'*Z{blockID}, b4{blockID});
            end
            h2 = tanh(h2);
            % step 5. h2 (D3 * [mbSize*L]) --> Y (D0 * [mbSize*L]), reconstruction 
            Y = 1./(1+exp(-bsxfun(@plus, W5'*h2, b5))); 
           
            % calculate the reconstruction error
            for i=1:mbSize
                id1 = (i-1)*L+1;
                id2 = i*L;
                X2(:,id1:id2) = repmat(X1(:,i),[1,L]);
            end
            energy = sum(sum(X2.*log(Y+1e-32) + (1-X2).*log(1-Y+1e-32)));
            if(rem(batchIdx,100)==0)
                fprintf('epoch %d, batch %d, ll is %d \n', epoch, batchIdx, energy/mbSize/L);
            end
            LL(epoch) = LL(epoch)+energy;
            
            %% backward propagation on hidden layers
            delta3 = X2-Y;
            delta2 = (W5*delta3).*(1-h2.^2);
            
            for blockID = 1:nBlocks
                delta1{blockID} = W4{blockID}*delta2;
                deltaMu{blockID} = deltaMu{blockID}*0;
                deltaAT{blockID} = deltaAT{blockID}*0;
                for i=1:mbSize
                    id1 = (i-1)*L+1;
                    id2 = i*L;
                    deltaAT{blockID}(:,:,i) = ...
                        delta1{blockID}(:,id1:id2)*transpose(epsilon{blockID}(:,id1:id2));
                    deltaMu{blockID}(:,i) = ...
                        sum(delta1{blockID}(:,id1:id2),2);
                end
                deltaAM{blockID} = reshape(deltaAT{blockID},[D2{blockID}^2, mbSize]);
            end
            delta0 = W2{1}*deltaMu{1} + W3{1}*deltaAM{1};
            for blockID = 2:nBlocks
                delta0 = delta0 + W2{blockID}*deltaMu{blockID} + W3{blockID}*deltaAM{blockID};
            end
            delta0 = delta0.*(1-h1.^2);
            
            %% backpropagation on parameters
            dW5 = h2*delta3'/mbSize/L + momentum*dW5;
            for blockID = 1:nBlocks
                dW4{blockID} = Z{blockID}*delta2'/mbSize/L + momentum*dW4{blockID};
                dW3{blockID} = h1*deltaAM{blockID}'/mbSize/L + momentum*dW3{blockID};
                dW2{blockID} = h1*deltaMu{blockID}'/mbSize/L + momentum*dW2{blockID};
            end
            dW1 = X1*delta0'/mbSize/L + momentum*dW1;

            db5 = sum(delta3,2)/mbSize/L + momentum*db5;
            for blockID = 1:nBlocks
                db4{blockID} = sum(delta2,2)/mbSize/L + momentum*db4{blockID};
                db3{blockID} = sum(deltaAM{blockID},2)/mbSize/L + momentum*db3{blockID};
                db2{blockID} = sum(deltaMu{blockID},2)/mbSize/L + momentum*db2{blockID};
            end
            db1 = sum(delta0,2)/mbSize/L + momentum*db1;

            W5 = W5 + lrate.*dW5; b5 = b5 + lrate.*db5;
            for blockID = 1:nBlocks
                W4{blockID} = W4{blockID} + lrate.*dW4{blockID}; 
                b4{blockID} = b4{blockID} + lrate.*db4{blockID};
                W3{blockID} = W3{blockID} + lrate.*dW3{blockID}; 
                b3{blockID} = b3{blockID} + lrate.*db3{blockID};
                W2{blockID} = W2{blockID} + lrate.*dW2{blockID}; 
                b2{blockID} = b2{blockID} + lrate.*db2{blockID};
            end
            W1 = W1 + lrate.*dW1; b1 = b1 + lrate.*db1;
            
        end
        fprintf('epoch %d, log-likelihood is %f\n', epoch, LL(epoch)/nSamples/L);
    end
    model{hid}.W5 = W5; model{hid}.b5 = b5;
    for blockID = 1:nBlocks
        model{hid}.W4{blockID} = W4{blockID}; model{hid}.b4{blockID} = b4{blockID};
        model{hid}.W3{blockID} = W3{blockID}; model{hid}.b3{blockID} = b3{blockID};
        model{hid}.W2{blockID} = W2{blockID}; model{hid}.b2{blockID} = b2{blockID};
    end
    model{hid}.W1 = W1; model{hid}.b1 = b1;
    model{hid}.LL = LL/nSamples/L;
end

matname = ['MNIST' num2str(nBlocks) 'blocks.mat'];
save(matname,'model');
