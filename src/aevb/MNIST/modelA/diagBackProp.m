function NN = diagBackProp(NN, NNsetting)

    L = NNsetting.L;
    batchSize = size(NN.X1,2);

    % architecture of NN
    %   X1 -NN.W1-> (Z1, h1) -NN.W2,NN.W3-> (mu, sigma, Z) -NN.W4-> (Z2, h2) -NN.W5-> (Z3,Y)
    % d(Loss) / d(Z3)
%     NN.delta3 = X2-Y;
    
    % d(Loss) / d(Z2)
    NN.delta2 = (NN.W5*NN.delta3).*(1-NN.h2).*NN.h2;

    % d(Loss) / d(mu), d(Loss) / d(sigma)
    NN.delta1a = NN.W4*NN.delta2;                % d/d(Mu)
    NN.delta1b = (NN.W4*NN.delta2).*NN.epsilon;     % d/d(sigma)

    NN.deltaMu = NN.deltaMu*0;
    NN.deltaSigma = NN.deltaSigma*0;
    for i=1:batchSize
        id1 = (i-1)*NNsetting.L+1;
        id2 = i*NNsetting.L;
        NN.deltaMu(:,i) = sum(NN.delta1a(:,id1:id2),2);
        NN.deltaSigma(:,i) = sum(NN.delta1b(:,id1:id2),2);
    end
    
    % KL divergence part: (need update)
    % d(KL) / d(mu), d(KL) / d(sigma)
    % be careful with "+" or "-" ahead of KL term
    NN.deltaMu = NN.deltaMu-L*NN.mu;
    NN.deltaSigma = NN.deltaSigma - L*(NN.sigma - NN.sigma.^(-1));
    NN.deltaBeta = NN.deltaSigma.*exp(0.5*NN.beta)/2;
    
    % d(Loss) / d(Z1)
    NN.delta0 = (NN.W2*NN.deltaMu + NN.W3*NN.deltaBeta).*(1-NN.h1).*NN.h1;
    
    %% backpropagation on parameters
    NN.dW5 = NN.h2*NN.delta3'/batchSize/L + NN.momentum*NN.dW5;
    NN.dW4 = NN.Z*NN.delta2'/batchSize/L + NN.momentum*NN.dW4;
    NN.dW3 = NN.h1*NN.deltaBeta'/batchSize/L + NN.momentum*NN.dW3;
    NN.dW2 = NN.h1*NN.deltaMu'/batchSize/L + NN.momentum*NN.dW2;
    NN.dW1 = NN.X1*NN.delta0'/batchSize/L + NN.momentum*NN.dW1;
    
    NN.db5 = sum(NN.delta3,2)/batchSize/L + NN.momentum*NN.db5;
    NN.db4 = sum(NN.delta2,2)/batchSize/L + NN.momentum*NN.db4;
    NN.db3 = sum(NN.deltaBeta,2)/batchSize/L + NN.momentum*NN.db3;
    NN.db2 = sum(NN.deltaMu,2)/batchSize/L + NN.momentum*NN.db2;
    NN.db1 = sum(NN.delta0,2)/batchSize/L + NN.momentum*NN.db1;
    
end
