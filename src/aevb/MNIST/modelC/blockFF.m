function NN = blockFF(NN, setting, X1)

    NN.X1 = X1;
    batchSize = size(NN.X1,2);
    L = setting.L;
    nBlocks = NN.nBlocks;
    mbSize = size(X1,2);
    
    NN.h1 = sigmoid(bsxfun(@plus, NN.W1'*NN.X1, NN.b1));
    
    for blockID = 1:nBlocks
        NN.Mu{blockID} = bsxfun(@plus, NN.W2{blockID}'*NN.h1, NN.b2{blockID});
        NN.Beta{blockID} = bsxfun(@plus, NN.W3{blockID}'*NN.h1, NN.b3{blockID});
        NN.AM{blockID} = exp(NN.Beta{blockID});
        NN.AT{blockID} = reshape(NN.AM{blockID}, [NN.D2, NN.D2, batchSize]);
        NN.epsilon{blockID} = randn(NN.D2, batchSize*L);
        
        for i=1:batchSize
            id1 = (i-1)*L+1;
            id2 = i*L;
            NN.Z{blockID}(:, id1:id2) = bsxfun(@plus, NN.AT{blockID}(:,:,i)*NN.epsilon{blockID}(:,id1:id2), NN.Mu{blockID}(:,i));
        end
    end

    NN.h2 = bsxfun(@plus, NN.W4{1}'*NN.Z{1}, NN.b4);
    for blockID = 2:nBlocks
        NN.h2 = NN.h2 + NN.W4{blockID}'*NN.Z{blockID};
    end
    NN.h2 = 1./(1+exp(-NN.h2));

    NN.Y = sigmoid(bsxfun(@plus, NN.W5'*NN.h2, NN.b5));

    for i=1:mbSize
        id1 = (i-1)*L+1;
        id2 = i*L;
        X2(:,id1:id2) = repmat(X1(:,i),[1,L]);
    end
    
    %% calculate the reconstruction loss
    X2 = zeros(NN.D0, size(NN.X1,2)*L);
    for i=1:batchSize
        id1 = (i-1)*L+1;
        id2 = i*L;
        X2(:,id1:id2) = repmat(NN.X1(:,i),[1,L]);
    end
    
    NN.loss = -sum(sum(X2.*log(NN.Y+1e-32) + (1-X2).*log(1-NN.Y+1e-32)));
    NN.delta3 = X2-NN.Y;
end

function y = sigmoid(x)
    y = 1./(1+exp(-x));
end
