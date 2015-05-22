function saveModel(name, NN, process)

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
    
    model.prior = NN.prior;
    % properties
    model.D0 = NN.D0;
    model.D1 = NN.D1;
    model.D2 = NN.D2;
    model.D3 = NN.D3;
    model.data = 'face';
    
    model.shape = NN.shape;
    model.alg = NN.alg;
    model.algInit = NN.algInit;
    model.ftLoss = NN.ftLoss;

    if(strcmp(model.shape, 'block'))
        model.nBlocks = NN.nBlocks;
        model.blocks = NN.blocks;
    end
    
    % learning hyperparameters
    if(strcmp(NN.alg, 'adadelta'))
        model.rho = NN.rho;
        model.const = NN.const;
        model.momentum = NN.momentum;
    elseif(strcmp(NN.alg, 'sgd'))
        model.lrat = NN.lrate;
        model.momentum = NN.momentum;
    end
    
    %% need revision to save the learning parameters
    
    if(nargin==2)
        save(name, 'model');
    elseif(nargin==3)
        save(name, 'model', 'process');
    end
end
