function NN = defaultNNinit(NN, setting)
    
    % initialize the parameters
    NN.W1 = rand(NN.D0, NN.D1)-0.5;  NN.b1 = rand(NN.D1, 1);
    NN.W2 = rand(NN.D1, NN.D2)-0.5;  NN.b2 = rand(NN.D2, 1);
    NN.W3 = zeros(NN.D1, NN.D2);     NN.b3 = zeros(NN.D2,1);  % question: if we initialize W randomly, will there be numerical stability issue?
    NN.W4 = rand(NN.D2, NN.D3)-0.5;  NN.b4 = rand(NN.D3, 1);
    NN.W5 = rand(NN.D3, NN.D0)-0.5;  NN.b5 = rand(NN.D0, 1);
    
    NN.dW5 = NN.W5*0; NN.db5 = NN.b5*0;
    NN.dW4 = NN.W4*0; NN.db4 = NN.b4*0;
    NN.dW3 = NN.W3*0; NN.db3 = NN.b3*0;
    NN.dW2 = NN.W2*0; NN.db2 = NN.b2*0;
    NN.dW1 = NN.W1*0; NN.db1 = NN.b1*0;
    
    NN.delta3      = zeros(NN.D0, setting.mbSize*setting.L);
    NN.delta2      = zeros(NN.D3, setting.mbSize*setting.L);
    NN.delta1a     = zeros(NN.D2, setting.mbSize*setting.L);
    NN.delta1b     = zeros(NN.D2, setting.mbSize*setting.L);
    NN.deltaMu     = zeros(NN.D2, setting.mbSize);
    NN.deltaSigma  = zeros(NN.D2, setting.mbSize);
    NN.deltaBeta   = zeros(NN.D2, setting.mbSize);
    NN.delta0      = zeros(NN.D1, setting.mbSize);
    
end