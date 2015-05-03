% pretrain the dimensionality reduction part of the model
% input and output is h layer, intermediate layer is (mu, sigma, Z)

load ../../data/faceData1K.mat
dataTr = hiddenFace;
clear hiddenFace;

D0 = size(dataTr,1);    % 1K
D1 = 3;               % mu, simga and Z layer(s)
L = 10;                 % L copies of samples ~ Z_{l}|X 
mbSize = 50;

% structure: X1-->h1-->(mu, beta)-->Z-->h2-->Y
X1      = zeros(D0, mbSize);
mu      = zeros(D1, mbSize);
beta	= zeros(D1, mbSize);        % std 
lambda 	= zeros(D1, mbSize);        % variance 
Z       = zeros(D1, mbSize*L);
Y       = zeros(D0, mbSize*L);      % reconstruction
X2      = zeros(D0, mbSize*L);      % used for error estimation

% learning hyperparameters: {lrate, momentum]
sgdParams = [
    1/1000 0.9;
    1/2000 0.9;
    1/3000 0.9;
    1/4000 0.9;
    1/1000 0.5;
    1/2000 0.5;
    1/3000 0.5;
    1/4000 0.5];

for hid = 2:2%1:size(sgdParams,1);
    lrate = sgdParams(hid,1);
    momentum = sgdParams(hid,2);
    
    % initialize the parameters
    % there are at least TWO ways to initialize the parameters
    % 1. random intialization
    % 2. use PCA basis parameters to initialize the projection [X1 --> mu]
    W1 = rand(D0, D1)-0.5;  b1 = rand(D1, 1);
    W2 = rand(D0, D1)-0.5;  b2 = rand(D1, 1);
    W3 = rand(D1, D0)-0.5;  b3 = rand(D0, 1);
    
    dW1 = W1*0; db1 = b1*0;
    dW2 = W2*0; db2 = b2*0;
    dW3 = W3*0; db3 = b3*0;

    nSamples = size(dataTr,2);
    numBatch = floor(nSamples/mbSize);

    % gradient w.r.t.
    % . H
    % . mu
    % . beta
    % architecture: X1 --> {mu, beta, lmabda} ==> Z --> H --> Y
    deltaH      = zeros(D1, mbSize*L);
    deltaMu     = zeros(D1, mbSize);
    deltaBeta   = zeros(D1, mbSize);
    
    nEpoch = 30;
    LL = zeros(nEpoch,1);
    for epoch = 1:nEpoch
        index = randperm(nSamples);
        for batchIdx = 1:numBatch
            firstIdx = (batchIdx-1)*mbSize+1;
            lastIdx = batchIdx*mbSize;
            
            X1 = dataTr(:, index(firstIdx:lastIdx));

            mu = bsxfun(@plus, W1'*X1, b1);
            beta = bsxfun(@plus, W2'*X1, b2);
            lambda = exp(0.5*beta);
            % lambda: STD
            % beta: intermediate variables
            % beta:=\log(lambda^2)
            % lambda:= exp(beta/2)
            
            epsilon = randn(D1, mbSize*L);
            % sampling:
            for i=1:mbSize
                id1 = (i-1)*L+1;
                id2 = i*L;
                Z(:, id1:id2) = bsxfun(@plus, diag(lambda(:,i))*epsilon(:,id1:id2), mu(:,i));
            end

            % step 5. h2 (D3 * [mbSize*L]) --> Y (D0 * [mbSize*L]), reconstruction 
            Y = 1./(1+exp(-bsxfun(@plus, W3'*Z, b3)));
            
            % calculate the reconstruction error
            for i=1:mbSize
                id1 = (i-1)*L+1;
                id2 = i*L;
                X2(:,id1:id2) = repmat(X1(:,i),[1,L]);
            end
            energy = sum(sum(X2.*log(Y+1e-16) + (1-X2).*log(1-Y+1e-16)));
            
%             if(rem(batchIdx,100)==0)
%                 fprintf('epoch %d, batch %d, ll is %d \n', epoch, batchIdx, energy/mbSize/L);
%             end
            LL(epoch) = LL(epoch)+energy; % log-probability of epoch of data

            %% backward propagation on layers
            deltaH = X2-Y;

            deltaZ = W3*deltaH;
            
            deltaMu = deltaMu*0;
            deltaBeta = deltaBeta*0;
            for i=1:mbSize
                id1 = (i-1)*L+1;
                id2 = i*L;
                deltaMu(:,i) = sum(deltaZ(:,id1:id2),2);
                deltaBeta(:,i) = 0.5*sum(deltaZ(:,id1:id2).*epsilon(:,id1:id2),2).*exp(0.5*beta(:,i));
            end
            
            % the following 2 rows incorporate the gradient from prior dist
            deltaMu = deltaMu-L*mu;
            deltaBeta = deltaBeta + 0.5*L*(1-lambda.^2);
            % the above 2 rows incorporate the gradient from prior dist
            
            %% backpropagation on parameters
            dW3 = Z*deltaH'/mbSize/L + momentum*dW3;
            dW2 = X1*deltaBeta'/mbSize/L + momentum*dW2;
            dW1 = X1*deltaMu'/mbSize/L + momentum*dW1;
            
            db3 = sum(deltaH,2)/mbSize/L + momentum*db3;
            db2 = sum(deltaBeta,2)/mbSize/L + momentum*db2;
            db1 = sum(deltaMu,2)/mbSize/L + momentum*db1;
            
            % SGD update parameters
            W3 = W3 + lrate.*dW3; b3 = b3 + lrate.*db3;
            W2 = W2 + lrate.*dW2; b2 = b2 + lrate.*db2;
            W1 = W1 + lrate.*dW1; b1 = b1 + lrate.*db1;
        end
        
        fprintf('epoch %d, log-likelihood is %f\n', epoch, LL(epoch)/nSamples/L);
    end
    model{hid}.W3 = W3; model{hid}.b3 = b3;
    model{hid}.W2 = W2; model{hid}.b2 = b2;
    model{hid}.W1 = W1; model{hid}.b1 = b1;
    model{hid}.LL = LL/nSamples/L;
    
%    save(['ptFixPriorDim' num2str(D1) '.mat'],'model','sgdParams');
end
