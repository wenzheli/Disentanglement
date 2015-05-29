function NN = blockFF(NN, setting, X1, Y1, supervised)
% the input Y1 is binary indicator matrix format

    NN.X1 = X1;
    batchSize = size(NN.X1,2);
    L = setting.L;
    nBlocks = NN.nBlocks;
    
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

    for classID = 1:NN.nClasses
        NN.predY{classID} = bsxfun(@plus, NN.Wc{classID}'*NN.Z{NN.mapBlock(classID)}, NN.bc{classID});
        NN.predY{classID} = exp(bsxfun(@minus, NN.predY{classID}, max(NN.predY{classID})));
        NN.predY{classID} = bsxfun(@rdivide, NN.predY{classID}, sum(NN.predY{classID}));
    end
    

    NN.h2 = bsxfun(@plus, NN.W4{1}'*NN.Z{1}, NN.b4);
    for blockID = 2:nBlocks
        NN.h2 = NN.h2 + NN.W4{blockID}'*NN.Z{blockID};
    end
    NN.h2 = 1./(1+exp(-NN.h2));

    NN.reconX = bsxfun(@plus, NN.W5'*NN.h2, NN.b5);

    idx = (0:(batchSize-1))*setting.L;
    for i=1:setting.L
        NN.delta3(:,idx+i) = X1;
        for classID = 1:NN.nClasses
            NN.deltaC{classID}(:,idx+i) = Y1{classID};
        end
    end

    % construct the true label
    NN.lossRecon = sum(sum((NN.delta3-NN.reconX).^2));
    NN.delta3 = NN.delta3-NN.reconX;
%     NN.lossPred = -sum(sum(NN.deltaC{m}.*log(NN.predY+1e-32) + (1-NN.deltaC{m}).*log(1-NN.predY)));
    supervised = transpose(reshape(repmat(supervised,[L,1]), batchSize*L,1));
    for classID = 1:NN.nClasses
        [~, l1{classID}] = max(NN.deltaC{classID}); % true classes
        [~, l2{classID}] = max(NN.predY{classID});  % pred classes
        NN.lossPred(classID) = sum((l1{classID}~=l2{classID}).*supervised);
        NN.deltaC{classID} = NN.cWeight(classID)*bsxfun(@times, (NN.deltaC{classID} - NN.predY{classID}), supervised);
        % a less efficient way of semi-supervised learning:
        %   multiply the derivative of unlabeled samples with 0
    end

%     keyboard % verify the new codes
end

function y = sigmoid(x)
    y = 1./(1+exp(-x));
end
