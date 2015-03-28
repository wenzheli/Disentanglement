load data/batchtraindata.mat;
dataTr = batchdata';
clear batchdata;

% load demoFrey;
% dataTr = dataTr/256;
mbSize = 50;
D0 = size(dataTr,1);
D1 = 200;

% structure
% X1 --tanh--> h1 --W--> {mu, sigma} --Gau--> X2
X1      = zeros(D0, mbSize);
h1      = zeros(D1, mbSize);
mu      = zeros(D1, mbSize);
sigma   = zeros(D0, mbSize);
X2      = zeros(D0, mbSize);

% initialize the parameters
W1 = rand(D0, D1)-0.5;  b1 = rand(D1, 1);
W2 = zeros(D1, D0)-0.5;  b2 = rand(D0, 1);
W3 = zeros(D1, D0)-0.5;  b3 = rand(D0, 1);

dW1 = W1*0; db1 = b1*0;
dW2 = W2*0; db2 = b2*0;
dW3 = W3*0; db3 = b3*0;

% if we use SGD: 
lrate = 0.01;
momentum = 0.9;

% if we use AdaGrad:
% lW3 = W3*0+1e-32;   lb3 = b3*0+1e-32;
% lW2 = W2*0+1e-32;   lb2 = b2*0+1e-32;
% lW1 = W1*0+1e-32;   lb1 = b1*0+1e-32;

nSamples = size(dataTr,2);
numBatch = floor(nSamples/mbSize);
index = randperm(nSamples);

nEpoch = 10;
LossTr = zeros(nEpoch,1);

for epoch = 1:nEpoch
    fprintf('learning epoch %d, ', epoch);
    for batchIdx = 1:numBatch
        firstIdx = (batchIdx-1)*mbSize+1;
        lastIdx = batchIdx*mbSize;
        X1 = dataTr(:, index(firstIdx:lastIdx));

        %% forward propagation 
        % step 1. X (D*mbSize) --> h1 (D1 * mbSize)
        h1 = tanh(bsxfun(@plus, W1'*X1, b1));

        % step 2. h1 (D1 * mbSize) --> mu (D2 * mbSize), A (D2^2 * mbSize)
        mu = bsxfun(@plus, W2'*h1, b2);
        sigma0 = bsxfun(@plus, W3'*h1, b3);
        sigma = exp(sigma0);
        
        % step 3. estimate the log-likelihood for verification only
        energy = - sum(sum((X1-mu).^2./(sigma+1e-32))) - sum(sum(log(sigma+1e-32)));
        LossTr(epoch) = LossTr(epoch) + energy;
        
        %% backward propagation, layer part
        
        % d(L)/d(sigma), of size (D0 * [mbSize*L])
        deltaMu = -0.5./sigma + 0.5*((X1-mu)./sigma).^2;
        % d(L)/d(mu), of size (D0 * [mbSize*L])
        deltaSigma = (X1-mu)./sigma;
        
        % d(L)/d(Z), of size (D3 * [mbSize*L])
        deltaZ = (W2*deltaMu + W3*deltaSigma).*(1-h1.^2);
        
        %% backward propagation, parameter part
        dW3 = h1*deltaSigma'/mbSize + momentum*dW3;
        dW2 = h1*deltaMu'/mbSize + momentum*dW2;
        dW1 = X1*deltaZ'/mbSize + momentum*dW1;
        
        db3 = sum(deltaSigma, 2) + momentum*db3;
        db2 = sum(deltaMu, 2) + momentum*db2;
        db1 = sum(deltaZ, 2) + momentum*db1;
        
        if(isnan(sum(b3(:))) || isinf(sum(b3(:))))
            keyboard
        end
        if(isnan(sum(b2(:))) || isinf(sum(b2(:))))
            keyboard
        end
        if(isnan(sum(b1(:))) || isinf(sum(b1(:))))
            keyboard
        end
        if(isnan(sum(W3(:))) || isinf(sum(W3(:))))
            keyboard
        end
        if(isnan(sum(W2(:))) || isinf(sum(W2(:))))
            keyboard
        end
        if(isnan(sum(W1(:))) || isinf(sum(W1(:))))
            keyboard
        end
        
        if(isnan(sum(db3(:))) || isinf(sum(db3(:))))
            keyboard
        end
        if(isnan(sum(db2(:))) || isinf(sum(db2(:))))
            keyboard
        end
        if(isnan(sum(db1(:))) || isinf(sum(db1(:))))
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
        
        %% update gradient using SGD
        W3 = W3+lrate*dW3;
        W2 = W2+lrate*dW2;
        W1 = W1+lrate*dW1;
        
        b3 = b3+lrate*db3;
        b2 = b2+lrate*db2;
        b1 = b1+lrate*db1;
        
        %% update gradient using AdaGrad
%         lW6 = sqrt(lW6.^2+dW6.^2);
%         lW5 = sqrt(lW5.^2+dW5.^2);
%         lW4 = sqrt(lW4.^2+dW4.^2);
%         lW3 = sqrt(lW3.^2+dW3.^2);
%         lW2 = sqrt(lW2.^2+dW2.^2);
%         lW1 = sqrt(lW1.^2+dW1.^2);
            
           
%         lb6 = sqrt(lb6.^2+db6.^2);
%         lb5 = sqrt(lb5.^2+db5.^2);
%         lb4 = sqrt(lb4.^2+db4.^2);
%         lb3 = sqrt(lb3.^2+db3.^2);
%         lb2 = sqrt(lb2.^2+db2.^2);
%         lb1 = sqrt(lb1.^2+db1.^2);       
        
%         if(~(epoch==1 && batchIdx==1))
%             W6 = W6 + dW6./lW6;
%             W5 = W5 + dW5./lW5;
%             W4 = W4 + dW4./lW4;
%             W3 = W3 + dW3./lW3;
%             W2 = W2 + dW2./lW2;
%             W1 = W1 + dW1./lW1;
% 
%             b6 = b6 + db6./lb6;
%             b5 = b5 + db5./lb5;
%             b4 = b4 + db4./lb4;
%             b3 = b3 + db3./lb3;
%             b2 = b2 + db2./lb2;
%             b1 = b1 + db1./lb1;
%         end

    end
    fprintf('log-likelihood is %f\n', LossTr(epoch));
end
