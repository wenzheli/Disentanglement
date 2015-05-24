function NN = fullBackProp(NN, NNsetting, BLK)
%   delta3
%   delta2
%   delta1
%   deltaMu
%   deltaAT
%   deltaAM
%   deltaBeta
%   delta0

    mbSize = size(NN.X1,2);
    L = NNsetting.L;
    
    NN.delta2 = (NN.W5*NN.delta3).*NN.h2.*(1-NN.h2);
    
    NN.delta1 = NN.W4*NN.delta2;
    NN.deltaMu = NN.deltaMu*0;
    NN.deltaAT = NN.deltaAT*0;
    for i=1:mbSize
        id1 = (i-1)*L+1;
        id2 = i*L;

        % gradients from reconstruction Log-Likelihood
        NN.deltaAT(:,:,i) = ...
            NN.delta1(:,id1:id2)*transpose(NN.epsilon(:,id1:id2))/L;
        NN.deltaMu(:,i) = ...
            sum(NN.delta1(:,id1:id2),2)/L;

        % gradients from minus KL divergence
        % w.r.t A: A^{-T} - Lambda*A
        % w.r.t mu: -Lambda*(mu - Emu)
        NN.deltaAT(:,:,i) = NN.deltaAT(:,:,i) + ...
            transpose(inv(NN.AT(:,:,i)+eye(NN.D2)*1e-32)) - BLK.Lambda*NN.AT(:,:,i);
        NN.deltaMu(:,i) = NN.deltaMu(:,i) - BLK.Lambda*(NN.Mu(:,i)-mean(BLK.Mu(:,id1:id2),2));
    end
    
    NN.deltaAM = reshape(NN.deltaAT,[NN.D2^2, mbSize]);
    NN.deltaBeta = NN.deltaAM.*NN.AM;

    NN.delta0 = (NN.W2*NN.deltaMu + NN.W3*NN.deltaBeta).*NN.h1.*(1-NN.h1);

    %% backpropagation - 2: w.r.t. parameters
    NN.dW5 = NN.h2*NN.delta3'/mbSize/L;
    NN.dW4 = NN.Z*NN.delta2'/mbSize/L;
    NN.dW3 = NN.h1*NN.deltaBeta'/mbSize;
    NN.dW2 = NN.h1*NN.deltaMu'/mbSize;
    NN.dW1 = NN.X1*NN.delta0'/mbSize;

    NN.db5 = sum(NN.delta3,2)/mbSize/L;
    NN.db4 = sum(NN.delta2,2)/mbSize/L;
    NN.db3 = sum(NN.deltaBeta,2)/mbSize;
    NN.db2 = sum(NN.deltaMu,2)/mbSize;
    NN.db1 = sum(NN.delta0,2)/mbSize;
end
