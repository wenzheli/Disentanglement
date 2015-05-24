function NN = fullFF(NN, setting, X1)%, X2)
    
    NN.X1 = X1;  
    % X1 will be used in backward propagation
    
    batchSize = size(NN.X1,2);
    L = setting.L;
    
    NN.h1 = 1./(1+exp(-bsxfun(@plus, NN.W1'*NN.X1, NN.b1)));
    
    NN.Mu = bsxfun(@plus, NN.W2'*NN.h1, NN.b2);
    NN.Beta = bsxfun(@plus, NN.W3'*NN.h1, NN.b3);
    NN.AM = exp(NN.Beta);
    NN.AT = reshape(NN.AM, [NN.D2, NN.D2, batchSize]);
    NN.epsilon = randn(NN.D2, batchSize*L);
    
    for i=1:batchSize
        id1 = (i-1)*L+1;
        id2 = i*L;
        NN.Z(:, id1:id2) = bsxfun(@plus, NN.AT(:,:,i)*NN.epsilon(:,id1:id2), NN.Mu(:,i));
    end
    
    NN.h2 = 1./(1+exp(-bsxfun(@plus, NN.W4'*NN.Z, NN.b4)));
    NN.Y = bsxfun(@plus, NN.W5'*NN.h2, NN.b5);
    
    %% calculate the reconstruction loss
    idx = L*(0:batchSize-1);
    X2 = zeros(NN.D0, batchSize*L);
    for i=1:L
        X2(:,idx+i) = X1;
    end
    
    NN.loss = sum(sum((X2-NN.Y).^2));
    NN.delta3 = X2-NN.Y;
end