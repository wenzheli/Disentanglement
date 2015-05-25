function saveModelA(name, NN)

    model.ftLoss = NN.ftLoss;
    model.algInit = NN.algInit;
    
    % parameters
    model.W1 = NN.W1;
    model.W2 = NN.W2;
    model.W3 = NN.W3;
    model.W4 = NN.W4;
    model.W5 = NN.W5;
    model.b1 = NN.b1;
    model.b2 = NN.b2;
    model.b3 = NN.b3;
    model.b4 = NN.b4;
    model.b5 = NN.b5;
    
    % properties
    model.D0 = NN.D0;
    model.D1 = NN.D1;
    model.D2 = NN.D2;
    model.D3 = NN.D3;
    model.data = NN.data;
    
    model.shape = NN.shape;
    model.alg = NN.alg;
    
    % learning hyperparameters
    if(strcmp(NN.alg, 'adadelta'))
        model.rho = NN.rho;
        model.const = NN.const;
        model.momentum = NN.momentum;
    elseif(strcmp(NN.alg, 'sgd'))
        model.lrat = NN.lrate;
        model.momentum = NN.momentum;
    end
    
    if(isfield(NN, 'ptZLoss'))
        model.ptZLoss = NN.ptZLoss;
    end
    
    if(isfield(NN, 'ptH2Loss'))
        model.ptH2Loss = NN.ptH2Loss;
    end
    
    save(name, 'model');

end
