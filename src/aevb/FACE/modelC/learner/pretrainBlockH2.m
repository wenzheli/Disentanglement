function [NN] = pretrainBlockH2(NN, NNsetting, dataX)
% 
    NN = initBlockH2(NN, NNsetting); 
    % actually this command is not necessary
    
    dataTr = genH2(NN, dataX);
    
    %% train a MLR
    
    nSamples = size(dataTr,2);
    mbSize = NNsetting.mbSize;
    numBatch = floor(nSamples/NNsetting.mbSize);
    NN.ptH2Loss = zeros(NNsetting.nEpoch,1);
    
    for epoch = 1:NNsetting.nEpoch
        index = randperm(nSamples);
        for batchIdx = 1:numBatch
            x = dataX(:, index((batchIdx-1)*mbSize+1:batchIdx*mbSize));
            h = dataTr(:, index((batchIdx-1)*mbSize+1:batchIdx*mbSize));
            y = bsxfun(@plus, NN.W5'*h, NN.b5);
            NN.ptH2Loss(epoch) = NN.ptH2Loss(epoch)+sum(sum((y-x).^2));
            NN.delta3 = x-y;
            
            NN.dW5 = h*NN.delta3'/mbSize + NN.momentum*NN.dW5;
            NN.db5 = sum(NN.delta3,2)/mbSize + NN.momentum*NN.db5;
            NN = updateBlockH2(NN, NNsetting);
            
            if(rem(batchIdx,100)==1)
                fprintf(2,'epoch %d, minibatch %d, recon loss: %f\n', ...
                    epoch, batchIdx, NN.ptH2Loss(epoch)/batchIdx/NNsetting.mbSize/NNsetting.L);
            end
        end
    end
end

function NN = initBlockH2(NN, setting)

    NN.W5 = (rand(NN.D3, NN.D0)-0.5)/10;
    NN.b5 = (rand(NN.D0, 1)-0.5)/10;
    NN.delta3 = zeros(NN.D0, setting.mbSize*setting.L);
    NN.dW5 = NN.W5*0; NN.db5 = NN.b5*0;
    
    % extra variables if use AdaDelta to optimize the parameters
    if(strcmp(setting.alg,'adadelta'))
        NN.dW5E = NN.W5*0;      NN.db5E = NN.b5*0;
        NN.deltaW5 = NN.W5*0;   NN.deltab5 = NN.b5*0;
        NN.deltaW5E = NN.W5*0;  NN.deltab5E = NN.b5*0;
    end
end

function dataH2 = genH2(NN, dataX)
    nSamples = size(dataX,2);
    dataH2 = zeros(NN.D3, nSamples);
    for firstIdx = 1:1000:nSamples
        lastIdx = min(firstIdx+999, nSamples);
        h1 = 1./(1+exp(-bsxfun(@times, NN.W1'*dataX(:,firstIdx:lastIdx), NN.b1)));
        for blockID = 1:NN.nBlocks
            Z{blockID} = bsxfun(@plus, NN.W2{blockID}'*h1, NN.b2{blockID});
            dataH2(:,firstIdx:lastIdx) = ...
                dataH2(:,firstIdx:lastIdx) + NN.W4{blockID}'*Z{blockID};
        end
        dataH2(:,firstIdx:lastIdx) = ...
            1./(1+exp(-bsxfun(@plus, dataH2(:,firstIdx:lastIdx), NN.b4)));
    end
end

function NN = updateBlockH2(NN, setting)

    if(strcmp(setting.alg, 'sgd'))
        % SGD update parameters
        NN.W5 = NN.W5 + NN.lrate.*NN.dW5; NN.b5 = NN.b5 + NN.lrate.*NN.db5;
    elseif(strcmp(setting.alg, 'adadelta'))
        %% ADA update of parameters and statistics
        NN.dW5E = NN.rho*NN.dW5E + (1-NN.rho)*NN.dW5.^2;
        NN.db5E = NN.rho*NN.db5E + (1-NN.rho)*NN.db5.^2;

        lrateW5 = sqrt(NN.deltaW5E+NN.const)./sqrt(NN.dW5E+NN.const);
        lrateb5 = sqrt(NN.deltab5E+NN.const)./sqrt(NN.db5E+NN.const);

        NN.deltaW5 = lrateW5.*NN.dW5; 
        NN.deltab5 = lrateb5.*NN.db5;

        NN.deltaW5E = NN.rho*NN.deltaW5E + (1-NN.rho)*NN.deltaW5.^2;
        NN.deltab5E = NN.rho*NN.deltab5E + (1-NN.rho)*NN.deltab5.^2;

        NN.W5 = NN.W5+NN.deltaW5;    NN.b5 = NN.b5+NN.deltab5;
    elseif(strcmp(setting.alg, 'adam'))
    end
end
