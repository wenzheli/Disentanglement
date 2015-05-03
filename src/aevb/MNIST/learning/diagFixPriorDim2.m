% aevb model
% on MNIST data
% 2-dimensional hidden space
% diagonal posterior model
% no prior distribution

load ../data/batchtraindata.mat;
dataTr = batchdata';
clear batchdata;

D0 = size(dataTr,1);    % 784
D1 = 200;               % h1
D2 = 2;                 % Z
D3 = 200;               % h2
L = 10;                 % L copies of samples ~ Z_{l}|X 
mbSize = 50;

% structure: X1-->h1-->(mu, beta)-->Z-->h2-->Y
X1      = zeros(D0, mbSize);
h1      = zeros(D1, mbSize);
mu      = zeros(D2, mbSize);
beta	= zeros(D2, mbSize);
lambda 	= zeros(D2, mbSize);
Z       = zeros(D2, mbSize*L);
h2      = zeros(D3, mbSize*L);
Y       = zeros(D0, mbSize*L);
X2      = zeros(D0, mbSize*L);


% learning hyperparameters: {lrate, momentum]
sgdParams = [
    1/1000 0.9;
    1/500 0.9;
    1/250 0.5];

for hid = 1:size(sgdParams,1);
    lrate = sgdParams(hid,1);
    momentum = sgdParams(hid,2);

    % initialize the parameters
    W1 = rand(D0, D1)-0.5;  b1 = rand(D1, 1);
    W2 = rand(D1, D2)-0.5;  b2 = rand(D2, 1);
    W3 = zeros(D1, D2);     b3 = zeros(D2,1);  % question: if we initialize W randomly, will there be numerical stability issue?
    W4 = rand(D2, D3)-0.5;  b4 = rand(D3, 1);
    W5 = rand(D3, D0)-0.5;  b5 = rand(D0, 1);
    W6 = rand(D3, D0)-0.5;  b6 = rand(D0, 1);

    dW6 = W6*0; db6 = b6*0;
    dW5 = W5*0; db5 = b5*0;
    dW4 = W4*0; db4 = b4*0;
    dW3 = W3*0; db3 = b3*0;
    dW2 = W2*0; db2 = b2*0;
    dW1 = W1*0; db1 = b1*0;

    nSamples = size(dataTr,2);
    numBatch = floor(nSamples/mbSize);

    delta3      = zeros(D0, mbSize*L);
    delta2      = zeros(D3, mbSize*L);
    delta1a     = zeros(D2, mbSize*L);
    delta1b     = zeros(D2, mbSize*L);
    deltaMu     = zeros(D2, mbSize);
    deltaBeta   = zeros(D2, mbSize);
    delta0      = zeros(D1, mbSize);

    nEpoch = 20;
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
            mu = bsxfun(@plus, W2'*h1, b2);
            beta = bsxfun(@plus, W3'*h1, b3);
            % note: beta:=\log(lambda^2) ==> std, lambda = exp(beta/2);
            lambda = exp(0.5*beta);
            
            % step 3. mu (D2 * mbSize), sigma(D2 * mbSize) --sampling--> Z (D2 * [mbSize*L])
            %Z = mu + sigma*epsilon; 
            epsilon = randn(D2, mbSize*L);
            % sampling:
            for i=1:mbSize
                id1 = (i-1)*L+1;
                id2 = i*L;
                %Z(:, id1:id2) = bsxfun(@plus, diag(exp(0.5*beta(:,i)))*epsilon(:,id1:id2), mu(:,i));
                Z(:, id1:id2) = bsxfun(@plus, diag(lambda(:,i))*epsilon(:,id1:id2), mu(:,i));
            end

            % step 4. Z (D2 *[mbSize*L]) --> h2(D3 * [mbSize*L])
            h2 = tanh(bsxfun(@plus, W4'*Z, b4));

            % step 5. h2 (D3 * [mbSize*L]) --> Y (D0 * [mbSize*L]), reconstruction 
            Y = 1./(1+exp(-bsxfun(@plus, W5'*h2, b5)));

            % calculate the reconstruction error
            for i=1:mbSize
                id1 = (i-1)*L+1;
                id2 = i*L;
                X2(:,id1:id2) = repmat(X1(:,i),[1,L]);
            end
            energy = sum(sum(X2.*log(Y+1e-32) + (1-X2).*log(1-Y+1e-32))); % log-likelihood of minibatch of data
            if(rem(batchIdx,100)==0)
                fprintf('epoch %d, batch %d, ll is %d \n', epoch, batchIdx, energy/mbSize/L);
            end
            LL(epoch) = LL(epoch)+energy; % log-probability of epoch of data

            %% backward propagation on layers
            delta3 = X2-Y;
            delta2 = (W5*delta3).*(1-h2.^2);

            % gradient from log-likelihood
            delta1a = W4*delta2;                % d/d(Mu)
            delta1b = (W4*delta2).*epsilon;     % d/d(Lambda)
            
            deltaMu = deltaMu*0;
            deltaBeta = deltaBeta*0;
            for i=1:mbSize
                id1 = (i-1)*L+1;
                id2 = i*L;
                deltaMu(:,i) = sum(delta1a(:,id1:id2),2);
                deltaBeta(:,i) = 0.5*sum(delta1b(:,id1:id2),2).*exp(0.5*beta(:,i));
            end
            
            % the following 2 rows incorporate the gradient from prior dist
            deltaMu = deltaMu-L*mu;
            deltaBeta = deltaBeta + 0.5*L*(1-lambda.^2);
            % the above 2 rows incorporate the gradient from prior dist
            
            delta0 = (W2*deltaMu + W3*deltaBeta).*(1-h1.^2);

            %% backpropagation on parameters
            dW5 = h2*delta3'/mbSize/L + momentum*dW5;
            dW4 = Z*delta2'/mbSize/L + momentum*dW4;
            dW3 = h1*deltaBeta'/mbSize/L + momentum*dW3;
            dW2 = h1*deltaMu'/mbSize/L + momentum*dW2;
            dW1 = X1*delta0'/mbSize/L + momentum*dW1;

            db5 = sum(delta3,2)/mbSize/L + momentum*db5;
            db4 = sum(delta2,2)/mbSize/L + momentum*db4;
            db3 = sum(deltaBeta,2)/mbSize/L + momentum*db3;
            db2 = sum(deltaMu,2)/mbSize/L + momentum*db2;
            db1 = sum(delta0,2)/mbSize/L + momentum*db1;

            % SGD update parameters
            W5 = W5 + lrate.*dW5; b5 = b5 + lrate.*db5;
            W4 = W4 + lrate.*dW4; b4 = b4 + lrate.*db4;
            W3 = W3 + lrate.*dW3; b3 = b3 + lrate.*db3;
            W2 = W2 + lrate.*dW2; b2 = b2 + lrate.*db2;
            W1 = W1 + lrate.*dW1; b1 = b1 + lrate.*db1;

        end
        fprintf('epoch %d, log-likelihood is %f\n', epoch, LL(epoch)/nSamples/L);
    end
    model{hid}.W5 = W5; model{hid}.b5 = b5;
    model{hid}.W4 = W4; model{hid}.b4 = b4;
    model{hid}.W3 = W3; model{hid}.b3 = b3;
    model{hid}.W2 = W2; model{hid}.b2 = b2;
    model{hid}.W1 = W1; model{hid}.b1 = b1;
    model{hid}.LL = LL/nSamples/L;
save('aevbDiagFixPriorDim2.mat','model','sgdParams');
end

