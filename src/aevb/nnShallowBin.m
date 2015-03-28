% shallow, binary observation and bernoulli hidden layer
% that's, auto-encoder
% load data/batchtraindata.mat;
% dataTr = batchdata';
% clear batchdata;

load data/demoFrey;
dataTr = dataTr/256;
mbSize = 50;
D0 = size(dataTr,1);
D1 = 200;

% structure
% X1 --tanh--> h1 --W--> {mu, sigma} --Gau--> X2
X1      = zeros(D0, mbSize);
h1      = zeros(D1, mbSize);
X2      = zeros(D0, mbSize);

% initialize the parameters
W1 = rand(D0, D1)-0.5;  b1 = rand(D1, 1);
W2 = rand(D1, D0)-0.5;  b2 = rand(D0, 1);

dW1 = W1*0; db1 = b1*0;
dW2 = W2*0; db2 = b2*0;

% if we use SGD: 
lrate = 0.01;
momentum1 = 0.5;
momentum2 = 0.5;%9;

% if we use AdaGrad:
% lW3 = W3*0+1e-32;   lb3 = b3*0+1e-32;
% lW2 = W2*0+1e-32;   lb2 = b2*0+1e-32;
% lW1 = W1*0+1e-32;   lb1 = b1*0+1e-32;

nSamples = size(dataTr,2);
numBatch = floor(nSamples/mbSize);
index = randperm(nSamples);

nEpoch = 400;
LossTr = zeros(nEpoch,1);

for epoch = 1:nEpoch
    if(epoch<=15)
        momentum = momentum1;
    else
        momentum = momentum2;
    end
    fprintf('learning epoch %d, \n', epoch);
    for batchIdx = 1:numBatch
        firstIdx = (batchIdx-1)*mbSize+1;
        lastIdx = batchIdx*mbSize;
        
        X0 = dataTr(:, index(firstIdx:lastIdx));
        X1 = X0.*(rand(size(X0))>0.05); % corrupting features
        %% forward propagation 
        % step 1. X (D * mbSize) --> h1 (D1 * mbSize)
        % h1 = tanh(bsxfun(@plus, W1'*X1, b1));
        h1 = 1./(1+exp(-bsxfun(@plus, W1'*X1, b1)));

        % step 2. h1 (D1 * mbSize) --> X (D * mbSize)
        X2 = 1./(1+exp(-bsxfun(@plus, W2'*h1, b2)));
        
        % step 3. estimate the log-likelihood for verification only
        energy = sum(sum(X1.*log(X2+1e-32) + (1-X1).*log(1-X2+1e-32)));
        LossTr(epoch) = LossTr(epoch) + energy;
        if(rem(batchIdx,100)==0)
            fprintf('\t reconstruction log-likelihood: %f\n', LossTr(epoch)/batchIdx/mbSize);
        end
        
        %% backward propagation, layer part
        
        % d(L)/d(pre_Y)
        deltaY = X0-X2;
        
        % d(L)/d(pre_h)
        % deltaH = (W2*deltaY).*(1-h1.^2);
        deltaH = (W2*deltaY).*(1-h1).*h1;
        
        %% backward propagation, parameter part
        dW2 = h1*deltaY'/mbSize + momentum*dW2;
        dW1 = X1*deltaH'/mbSize + momentum*dW1;
        
        db2 = sum(deltaY, 2) + momentum*db2;
        db1 = sum(deltaH, 2) + momentum*db1;
                
        if(isnan(sum(db2(:))) || isinf(sum(db2(:))))
            keyboard
        end
        if(isnan(sum(db1(:))) || isinf(sum(db1(:))))
            keyboard
        end
        if(isnan(sum(dW2(:))) || isinf(sum(dW2(:))))
            keyboard
        end
        if(isnan(sum(dW1(:))) || isinf(sum(dW1(:))))
            keyboard
        end
        
        %% update gradient using SGD
        W2 = W2+lrate*dW2;
        W1 = W1+lrate*dW1;
        
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
    fprintf('log-likelihood is %f\n', LossTr(epoch)/nSamples);
end
