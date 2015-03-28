load demoFrey;
dataTr = dataTr/256;
mbSize = 50;
D0 = size(dataTr,1);
D1 = 200;
D2 = 100;
D3 = 200;
L = 1; % sample L copies of Z for each X

X1      = zeros(D0, mbSize);
h1      = zeros(D1, mbSize);
mu      = zeros(D2, mbSize);
A1      = zeros(D2^2, mbSize);
A2      = zeros(D2, D2, mbSize);
Z       = zeros(D2, mbSize*L);
h2      = zeros(D3, mbSize*L);
eta     = zeros(D0, mbSize*L);
sigma   = zeros(D0, mbSize*L);
X2      = zeros(D0, mbSize*L);

% initialize the parameters
W1 = rand(D0, D1)-0.5;  b1 = rand(D1, 1);
W2 = rand(D1, D2)-0.5;  b2 = rand(D2, 1);
W3 = rand(D1, D2^2)-0.5;b3 = rand(D2^2, 1);
W4 = rand(D2, D3)-0.5;  b4 = rand(D3, 1);
W5 = rand(D3, D0)-0.5;  b5 = rand(D0, 1);
W6 = rand(D3, D0)-0.5;  b6 = rand(D0, 1);

% will use AdaGrad to update learning rate
lW6 = W6*0+1e-32;   lb6 = b6*0+1e-32;
lW5 = W5*0+1e-32;   lb5 = b5*0+1e-32;
lW4 = W4*0+1e-32;   lb4 = b4*0+1e-32;
lW3 = W3*0+1e-32;   lb3 = b3*0+1e-32;
lW2 = W2*0+1e-32;   lb2 = b2*0+1e-32;
lW1 = W1*0+1e-32;   lb1 = b1*0+1e-32;

nSamples = size(dataTr,2);
mbSize = 50;
numBatch = floor(nSamples/mbSize);
index = randperm(nSamples);

nEpoch = 10;
LLverify = zeros(nEpoch,1);
for epoch = 1:nEpoch
    for batchIdx = 1:numBatch
        firstIdx = (batchIdx-1)*mbSize+1;
        lastIdx = batchIdx*mbSize;
        X1 = dataTr(:, index(firstIdx:lastIdx));

        %% forward propagation 
        % step 1. X (D*mbSize) --> h1 (D1 * mbSize)
        h1 = tanh(bsxfun(@plus, W1'*X1, b1));

        % step 2. h1 (D1 * mbSize) --> mu (D2 * mbSize), A (D2^2 * mbSize)
        mu = bsxfun(@plus, W2'*h1, b2);
        A1 = bsxfun(@plus, W3'*h1, b3);
        A2 = reshape(A2, [D2, D2, mbSize]);

        % step 3. mu (D2 * mbSize), A (D2^2 * mbSize) --> Z (D2 * [mbSize*L])
        %Z = mu + A*epsilon;  % need more consideration here 
        epsilon = randn(D2, mbSize*L);
        id = 1;
        for i=1:L
            for j=1:mbSize
                Z(:, id) = mu(:,j)+A2(:,:,j)*epsilon(:,id);
                id = id+1;
            end
        end

        % step 4. Z (D2 * [mbSize*L]) --> h2 (D3 * [mbSize*L])
        h2 = tanh(bsxfun(@plus, W4'*Z, b4));

        % step 5. h2 (D3 * [mbSize*L]) --> eta(D0 * [mbSize*L]), sigma(D0 * [mbSize*L])
        eta = bsxfun(@plus, W5'*h2, b5);
        sigma = exp(bsxfun(@plus, W6'*h2, b6));

        % step 6. eta(D0 * [mbSize*L]), sigma(D0 * [mbSize*L]) --> X2 (D0 * [mbSize*L])
        
        for l=1:L
            X2(:,(l-1)*mbSize+1:l*mbSize) = X1;
        end
        if(isnan(-sum(sum((X2-eta).^2./sigma)) - sum(sum(log(sigma+1e-32)))))
            keyboard
        end
        LLverify(epoch) = LLverify(epoch)-sum(sum((X2-eta).^2./sigma)) - sum(sum(log(sigma+1e-32)));
        
        %% backward propagation
        delta6 = zeros(D0, mbSize*L);
        delta5 = zeros(D0, mbSize*L);
        delta4 = zeros(D3, mbSize*L);
        delta3T = zeros(D2, D2, mbSize);
        delta3M = zeros(D2^2, mbSize);
        delta2 = zeros(D2, mbSize);
        
        % d(L)/d(sigma), of size (D0 * [mbSize*L])
        delta6 = -0.5./sigma + 0.5*((X2-eta)./sigma).^2;
        % d(L)/d(eta), of size (D0 * [mbSize*L])
        delta5 = (X2-eta)./sigma;
        
        % d(L)/d(h2), of size (D3 * [mbSize*L])
        delta4 = W5*delta6 + W6*delta5;

        % d(L)/d(Z), of size (D2 * [mbSize*L])
        deltaZ = W4*(delta4.*(1-h2.^2));
        
        % d(L)/d(A), of size (D2^2 * [mbSize*L])
        % d(L)/d(mu), of size (D2 * [mbSize*L])
        delta3T = delta3T*0;
        delta2 = delta2*0;
        id = 1;
        for i=1:L
            for j=1:mbSize
                delta3T(:,:,j) = delta3T(:,:,j) + deltaZ(:,id)*epsilon(:,id)';
                id = id+1;
            end
            delta2 = delta2 + deltaZ(:,(i-1)*mbSize+1:i*mbSize);
        end
        
        % d(L)/d(h1), of size (D1 * mbSize)
        delta3M = reshape(delta3T, [D2^2, mbSize]);
        delta1 = W2*delta2 + W3*delta3M;
        
        %% backpropagation 2
        dW6 = h2*transpose(delta6.*sigma);
        dW5 = h2*delta5';
        dW4 = Z*transpose(delta4.*(1-h2.^2));
        dW3 = h1*delta3M';
        dW2 = h1*delta2';
        dW1 = X1*delta1';

        db6 = sum(delta6.*sigma,2);
        db5 = sum(delta5,2);
        db4 = sum(delta4.*(1-h2.^2),2);
        db3 = sum(delta3M,2);
        db2 = sum(delta2,2);
        db1 = sum(delta1,2);
        
        if(isnan(sum(dW6(:))) || isinf(sum(dW6(:))))
            keyboard
        end
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
        
        keyboard
        if(~(epoch==1 && batchIdx==1))
            W6 = W6 + dW6./lW6;
            W5 = W5 + dW5./lW5;
            W4 = W4 + dW4./lW4;
            W3 = W3 + dW3./lW3;
            W2 = W2 + dW2./lW2;
            W1 = W1 + dW1./lW1;

            b6 = b6 + db6./lb6;
            b5 = b5 + db5./lb5;
            b4 = b4 + db4./lb4;
            b3 = b3 + db3./lb3;
            b2 = b2 + db2./lb2;
            b1 = b1 + db1./lb1;
        end
        
        %% update gradient
%         lW6 = sqrt(lW6.^2+dW6.^2);
%         lW5 = sqrt(lW5.^2+dW5.^2);
%         lW4 = sqrt(lW4.^2+dW4.^2);
%         lW3 = sqrt(lW3.^2+dW3.^2);
%         lW2 = sqrt(lW2.^2+dW2.^2);
%         lW1 = sqrt(lW1.^2+dW1.^2);

        lW6 = lW6*0+5000*mbSize*L;
        lW5 = lW5*0+5000*mbSize*L;
        lW4 = lW4*0+500*mbSize*L;
        lW3 = lW3*0+5000*mbSize*L;
        lW2 = lW2*0+5000*mbSize*L;
        lW1 = lW1*0+500*mbSize*L; 
        
           
%         lb6 = sqrt(lb6.^2+db6.^2);
%         lb5 = sqrt(lb5.^2+db5.^2);
%         lb4 = sqrt(lb4.^2+db4.^2);
%         lb3 = sqrt(lb3.^2+db3.^2);
%         lb2 = sqrt(lb2.^2+db2.^2);
%         lb1 = sqrt(lb1.^2+db1.^2);             
        
        lb6 = lb6*0+5000*mbSize*L;
        lb5 = lb5*0+5000*mbSize*L;
        lb4 = lb4*0+500*mbSize*L;
        lb3 = lb3*0+5000*mbSize*L;
        lb2 = lb2*0+5000*mbSize*L;
        lb1 = lb1*0+500*mbSize*L;
        
       
    end
    fprintf('epoch %d, log-likelihood is %f\n', epoch, LLverify(epoch));
end
