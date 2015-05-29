function saveModelC(name, NN, process)
% checkList
% 1. NN architecture and properties
%   * shape and sizes
%   * name of dataset
%   * learning/optimization algorithm
%   * initialization method
%   * learning hyperparameters
%   * nBlocks
%   
% 2. NN parameters
%   * W1~W5, b1~b5
%   * parameters in NW prior 
%   
% 3. Learning Process
%   * reconstruction error in pretraining
%   * reconstruction error in finetuning


% 1. NN architecture and properties
    model.shape = NN.shape;

    model.D0 = NN.D0;
    model.D1 = NN.D1;
    model.D2 = NN.D2;
    model.D3 = NN.D3;
    
    model.data = 'face';
    model.alg = NN.alg;
    model.algInit = NN.algInit;
    if(strcmp(NN.alg, 'adadelta'))
        model.rho = NN.rho;
        model.const = NN.const;
        model.momentum = NN.momentum;
    elseif(strcmp(NN.alg, 'sgd'))
        model.lrat = NN.lrate;
        model.momentum = NN.momentum;
    end
    
    if(strcmp(model.shape, 'block'))
        model.nBlocks = NN.nBlocks;
        model.blocks = NN.blocks;
    end

% 2. parameters
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
    
    %model.prior = NN.prior;
    
% 3. Learning Process
    model.ftLossRecon = NN.ftLossRecon;
    model.ftLossPred = NN.ftLossPred;
    if(isfield(NN, 'ptZLoss'))
        model.ptZLoss = NN.ptZLoss;
    end
    if(isfield(NN, 'ptH2Loss'))
        model.ptH2Loss = NN.ptH2Loss;
    end
    
    if(nargin==2)
        save(name, 'model');
    elseif(nargin==3)
        save(name, 'model', 'process');
    end
end
