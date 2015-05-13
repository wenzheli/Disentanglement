function NN = updateNN(NN, setting)
    if(strcmp(setting.alg, 'sgd'))
        % SGD update parameters
        NN.W5 = NN.W5 + NN.lrate.*NN.dW5; NN.b5 = NN.b5 + NN.lrate.*NN.db5;
        NN.W4 = NN.W4 + NN.lrate.*NN.dW4; NN.b4 = NN.b4 + NN.lrate.*NN.db4;
        NN.W3 = NN.W3 + NN.lrate.*NN.dW3; NN.b3 = NN.b3 + NN.lrate.*NN.db3;
        NN.W2 = NN.W2 + NN.lrate.*NN.dW2; NN.b2 = NN.b2 + NN.lrate.*NN.db2;
        NN.W1 = NN.W1 + NN.lrate.*NN.dW1; NN.b1 = NN.b1 + NN.lrate.*NN.db1;
    elseif(strcmp(setting.alg, 'adadelta'))
    elseif(strcmp(setting.alg, 'adam'))
    end
end