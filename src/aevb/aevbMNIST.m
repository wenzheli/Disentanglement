% the struction of the model
% q(z|x): dim-784 Xs --> dim-200 hu --> dim-2 Gaussian Z
% p(x|z): dim-2 Gaussian Z --> dim-200 hu --> dim-784 Xs

load data/batchtraindata.mat;
dataTr = batchdata';
clear batchdata;

D0 = size(dataTr,1); % 784
D1 = 200;            % h1
D2 = 2;              % Z
D3 = 200;            % h2
L = 10;               % L copies Z_{l}|X 
mbSize = 50;

X1      = zeros(D0, mbSize);
h1      = zeros(D1, mbSize);
mu      = zeros(D2, mbSize);
beta	= zeros(D2, mbSize);
Z       = zeros(D2, mbSize*L);
h2      = zeros(D3, mbSize*L);
eta     = zeros(D0, mbSize*L);
sigma   = zeros(D0, mbSize*L); 
    % comment: I guess AdaGrad is required to handle the scale of sigma
X2      = zeros(D0, mbSize*L);
Y       = zeros(D0, mbSize*L);

% initialize the parameters
W1 = rand(D0, D1)-0.5;  b1 = rand(D1, 1);
W2 = rand(D1, D2)-0.5;  b2 = rand(D2, 1);
W3 = zeros(D1, D2);     b3 = zeros(D2,1);
W4 = rand(D2, D3)-0.5;  b4 = rand(D3, 1);
W5 = rand(D3, D0)-0.5;  b5 = rand(D0, 1);
W6 = rand(D3, D0)-0.5;  b6 = rand(D0, 1);

dW6 = W6*0; db6 = b6*0;
dW5 = W5*0; db5 = b5*0;
dW4 = W4*0; db4 = b4*0;
dW3 = W3*0; db3 = b3*0;
dW2 = W2*0; db2 = b2*0;
dW1 = W1*0; db1 = b1*0;

% will use AdaGrad to update learning rate
lW6 = W6*0+1e-32;   lb6 = b6*0+1e-32;
lW5 = W5*0+1e-32;   lb5 = b5*0+1e-32;
lW4 = W4*0+1e-32;   lb4 = b4*0+1e-32;
lW3 = W3*0+1e-32;   lb3 = b3*0+1e-32;
lW2 = W2*0+1e-32;   lb2 = b2*0+1e-32;
lW1 = W1*0+1e-32;   lb1 = b1*0+1e-32;

nSamples = size(dataTr,2);
numBatch = floor(nSamples/mbSize);

delta3      = zeros(D0, mbSize*L);
delta2      = zeros(D3, mbSize*L);
delta1a     = zeros(D2, mbSize*L);
delta1b     = zeros(D2, mbSize*L);
deltaMu     = zeros(D2, mbSize);
deltaBeta   = zeros(D2, mbSize);
delta0      = zeros(D1, mbSize);

nEpoch = 10;
Loss = zeros(nEpoch,1);
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
        % this is \log(lambda^2)

        % step 3. mu (D2 * mbSize), A (D2^2 * mbSize) --> Z (D2 * [mbSize*L])
        %Z = mu + A*epsilon;  % need more consideration here 
        epsilon = randn(D2, mbSize*L);
        % sampling:
        for i=1:mbSize
            id1 = (i-1)*L+1;
            id2 = i*L;
            Z(:, id1:id2) = bsxfun(@plus, diag(exp(0.5*beta(:,i)))*epsilon(:,id1:id2), mu(:,i));
            % consider epsilon .* sigma, due to the exp() effect in sigma, 
            %   some elements of epsilon.*sigma could be exponentially large
            %   and a Gaussian var with very large std implies any number
            %   between -\infty and +\infty
            %   which would be disasters for reconstruction
        end
        
        h2 = tanh(bsxfun(@plus, W4'*Z, b4));

        Y = 1./(1+exp(-bsxfun(@plus, W5'*h2, b5))); 
        
        for i=1:mbSize
            id1 = (i-1)*L+1;
            id2 = i*L;
            X2(:,id1:id2) = repmat(X1(:,i),[1,L]);
        end
        
        
        energy = sum(sum(X2.*log(Y+1e-32) + (1-X2).*log(1-Y+1e-32)));
        if(rem(batchIdx,100)==0)
            fprintf('epoch %d, batch %d, ll is %d \n', epoch, batchIdx, energy/mbSize/L);
        end
        Loss(epoch) = Loss(epoch)+ sum(sum(X2.*log(Y+1e-32) + (1-X2).*log(1-Y+1e-32)));
        
        %% backward propagation
        
        
        delta3 = X2-Y;
        delta2 = (W5*delta3).*(1-h2.^2);
        
        delta1a = W4*delta2;
        delta1b = (W4*delta2).*epsilon; % d/d(Lambda)
        deltaMu = deltaMu*0;
        deltaBeta = deltaBeta*0;
        for i=1:mbSize
            id1 = (i-1)*L+1;
            id2 = i*L;
            deltaMu(:,i) = sum(delta1a(:,id1:id2),2);
            deltaBeta(:,i) = 0.5*sum(delta1b(:,id1:id2),2).*exp(0.5*beta(:,i));
        end
        
        delta0 = (W2*deltaMu + W3*deltaBeta).*(1-h1.^2);
        
        %% backpropagation 2
        dW5 = h2*delta3'/mbSize/L;
        dW4 = Z*delta2'/mbSize/L;
        dW3 = h1*deltaBeta'/mbSize/L;
        dW2 = h1*deltaMu'/mbSize/L;
        dW1 = X1*delta0'/mbSize/L;

        db5 = sum(delta3,2)/mbSize/L;
        db4 = sum(delta2,2)/mbSize/L;
        db3 = sum(deltaBeta,2)/mbSize/L;
        db2 = sum(deltaMu,2)/mbSize/L;
        db1 = sum(delta0,2)/mbSize/L;
        
        if(isnan(sum(dW5(:))) || isinf(sum(dW5(:))))
            keyboard
        end
        if(isnan(sum(dW4(:))) || isinf(sum(dW4(:))))
            keyboard
        end
        if(isnan(sum(dW3(:))) || isinf(sum(dW3(:))))
            keyboard
        end
        if(isnan(sum(dW2(:))) || isinf(sum(dW2(:))))
            keyboard
        end
        if(isnan(sum(dW1(:))) || isinf(sum(dW1(:))))
            keyboard
        end
        
        %% update gradient
%         lW6 = sqrt(lW6.^2+dW6.^2);
%         lW5 = sqrt(lW5.^2+dW5.^2);
%         lW4 = sqrt(lW4.^2+dW4.^2);
%         lW3 = sqrt(lW3.^2+dW3.^2);
%         lW2 = sqrt(lW2.^2+dW2.^2);
%         lW1 = sqrt(lW1.^2+dW1.^2);

        lrate = 1/500;
        lW5 = lW5*0+lrate;
        lW4 = lW4*0+lrate;
        lW3 = lW3*0+lrate;
        lW2 = lW2*0+lrate;
        lW1 = lW1*0+lrate;
        
%         lb6 = sqrt(lb6.^2+db6.^2);
%         lb5 = sqrt(lb5.^2+db5.^2);
%         lb4 = sqrt(lb4.^2+db4.^2);
%         lb3 = sqrt(lb3.^2+db3.^2);
%         lb2 = sqrt(lb2.^2+db2.^2);
%         lb1 = sqrt(lb1.^2+db1.^2);
        lb5 = lb5*0+lrate;
        lb4 = lb4*0+lrate;
        lb3 = lb3*0+lrate;
        lb2 = lb2*0+lrate;
        lb1 = lb1*0+lrate;
        
        W5 = W5 + lW5.*dW5; b5 = b5 + lb5.*db5;
        W4 = W4 + lW4.*dW4; b4 = b4 + lb4.*db4;
        W3 = W3 + lW3.*dW3; b3 = b3 + lb3.*db3;
        W2 = W2 + lW2.*dW2; b2 = b2 + lb2.*db2;
        W1 = W1 + lW1.*dW1; b1 = b1 + lb1.*db1;
        
    end
    fprintf('epoch %d, log-likelihood is %f\n', epoch, Loss(epoch));
end
