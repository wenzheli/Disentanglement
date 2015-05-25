function NN = diagFF(NN, setting, X1)

    NN.X1 = X1;
    batchSize = size(NN.X1,2);
    
    %% forward propagation
    NN.h1 = sigmoid(bsxfun(@plus, NN.W1'*NN.X1, NN.b1));

    NN.mu = bsxfun(@plus, NN.W2'*NN.h1, NN.b2);
    NN.beta = bsxfun(@plus, NN.W3'*NN.h1, NN.b3);
    NN.sigma = exp(0.5*NN.beta);
    % note: covariance NN.sigma^2 = exp(NN.beta);
    
    % sampling:
    NN.epsilon = randn(NN.D2, batchSize*setting.L);
    NN.Sigmas = NN.epsilon*0;
    idx = (0:batchSize-1)*setting.L;
    for i=1:setting.L
        NN.Z(:,idx+i) = NN.mu;
        NN.Sigmas(:,idx+i) = NN.sigma;
    end
    NN.Z = NN.Z + NN.Sigmas.*NN.epsilon;
    
    sanityCheck = 0;
    if(sanityCheck)
        Z = NN.Z;
        for i=1:batchSize
            id1 = (i-1)*setting.L+1;
            id2 = i*setting.L;
            Z(:, id1:id2) = bsxfun(@plus, diag(NN.sigma(:,i))*NN.epsilon(:,id1:id2), NN.mu(:,i));
        end
        if(sum(sum(abs(Z-NN.Z)))>1e-10)
            error('incorrect assignment of values');
        end
    end

    NN.h2 = sigmoid(bsxfun(@plus, NN.W4'*NN.Z, NN.b4));
    NN.Y = (bsxfun(@plus, NN.W5'*NN.h2, NN.b5));
    %NN.Y = sigmoid(bsxfun(@plus, NN.W5'*NN.h2, NN.b5));
    
    %% calculate the reconstruction loss
    X2 = zeros(NN.D0, size(NN.X1,2)*setting.L);
    for i=1:batchSize
        id1 = (i-1)*setting.L+1;
        id2 = i*setting.L;
        X2(:,id1:id2) = repmat(NN.X1(:,i),[1,setting.L]);
    end
    
    NN.loss = sum(sum((X2-NN.Y).^2));
    NN.delta3 = X2-NN.Y;
end

function X = sigmoid(X)
    X = 1./(1+exp(-X));
end
