function NN = blockBackProp(NN, NNsetting, NW)
%   delta3 --> delta2 --> 
%       {delta1, deltaMu, deltaAT, deltaAM, deltaBeta} --> delta0

    mbSize = size(NN.X1,2);
    L = NNsetting.L;
    nBlocks = NN.nBlocks;
    
    NN.delta2 = (NN.W5*NN.delta3).*NN.h2.*(1-NN.h2);
    for blockID = 1:nBlocks
        NN.delta1{blockID} = NN.W4{blockID}*NN.delta2;
        NN.deltaMu{blockID} = NN.deltaMu{blockID}*0;
        NN.deltaAT{blockID} = NN.deltaAT{blockID}*0;
        for i=1:mbSize
            id1 = (i-1)*L+1;
            id2 = i*L;

            % gradients from reconstruction Log-Likelihood
            NN.deltaAT{blockID}(:,:,i) = ...
                NN.delta1{blockID}(:,id1:id2)*transpose(NN.epsilon{blockID}(:,id1:id2))/L;
            NN.deltaMu{blockID}(:,i) = ...
                sum(NN.delta1{blockID}(:,id1:id2),2)/L;

            % gradients from minus KL divergence
            % w.r.t A: A^{-T} - Lambda*A
            % w.r.t mu: -Lambda*(mu - Emu)
            NN.deltaAT{blockID}(:,:,i) = NN.deltaAT{blockID}(:,:,i) + ...
                transpose(inv(NN.AT{blockID}(:,:,i)+eye(NN.D2)*1e-32)) -... 
                NW.Lambda{blockID}*NN.AT{blockID}(:,:,i);
            NN.deltaMu{blockID}(:,i) = NN.deltaMu{blockID}(:,i) - ...
                NW.Lambda{blockID}*(NN.Mu{blockID}(:,i)-NW.mu{blockID});
        end
        NN.deltaAM{blockID} = reshape(NN.deltaAT{blockID},[NN.D2^2, mbSize]);
        NN.deltaBeta{blockID} = NN.deltaAM{blockID}.*NN.AM{blockID};
    end

    NN.delta0 = NN.W2{1}*NN.deltaMu{1} + NN.W3{1}*NN.deltaBeta{1};
    for blockID = 2:NN.nBlocks
        NN.delta0 = NN.delta0 + NN.W2{blockID}*NN.deltaMu{blockID} + NN.W3{blockID}*NN.deltaBeta{blockID};
    end
    NN.delta0 = NN.delta0.*NN.h1.*(1-NN.h1);

    %% backpropagation - 2: w.r.t. parameters
    NN.dW5 = NN.h2*NN.delta3'/mbSize/L;
    for blockID = 1:NN.nBlocks
        NN.dW4{blockID} = NN.Z{blockID}*NN.delta2'/mbSize/L;
        NN.dW3{blockID} = NN.h1*NN.deltaBeta{blockID}'/mbSize;
        NN.dW2{blockID} = NN.h1*NN.deltaMu{blockID}'/mbSize;
    end
    NN.dW1 = NN.X1*NN.delta0'/mbSize;

    NN.db5 = sum(NN.delta3,2)/mbSize/L;
    NN.db4 = sum(NN.delta2,2)/mbSize/L;
    for blockID = 1:NN.nBlocks
        NN.db3{blockID} = sum(NN.deltaBeta{blockID},2)/mbSize;
        NN.db2{blockID} = sum(NN.deltaMu{blockID},2)/mbSize;
    end
    NN.db1 = sum(NN.delta0,2)/mbSize;
end