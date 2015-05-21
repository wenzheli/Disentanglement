function NN = defaultNNinit(NN, setting)
% initialize the NN parameters RANDOMLY
    
    if(strcmp(NN.shape, 'diag'))
        NN.W1 = (rand(NN.D0, NN.D1)-0.5)/10;
        NN.b1 = (rand(NN.D1, 1)-0.5)/10;

        NN.W2 = (rand(NN.D1, NN.D2)-0.5)/10;
        NN.b2 = (rand(NN.D2,1)-0.5)/10;

        NN.W3 = (rand(NN.D1, NN.D2)-0.5)/10;
        NN.b3 = (rand(NN.D2,1)-0.5)/10;

        NN.W4 = (rand(NN.D2, NN.D3)-0.5)/10;
        NN.b4 = (rand(NN.D3, 1)-0.5)/10;

        NN.W5 = (rand(NN.D3, NN.D0)-0.5)/10;
        NN.b5 = (rand(NN.D0, 1)-0.5)/10;

        % intermediate variabls in backpropagation
        NN.delta3      = zeros(NN.D0, setting.mbSize*setting.L);
        NN.delta2      = zeros(NN.D3, setting.mbSize*setting.L);
        NN.delta1a     = zeros(NN.D2, setting.mbSize*setting.L);
        NN.delta1b     = zeros(NN.D2, setting.mbSize*setting.L);
        NN.deltaMu     = zeros(NN.D2, setting.mbSize);
        NN.deltaSigma  = zeros(NN.D2, setting.mbSize);
        NN.deltaBeta   = zeros(NN.D2, setting.mbSize);
        NN.delta0      = zeros(NN.D1, setting.mbSize);

        % gradient variables used in AdaDelta updating of parameters
        NN.dW5 = NN.W5*0; NN.db5 = NN.b5*0;
        NN.dW4 = NN.W4*0; NN.db4 = NN.b4*0;
        NN.dW3 = NN.W3*0; NN.db3 = NN.b3*0;
        NN.dW2 = NN.W2*0; NN.db2 = NN.b2*0;
        NN.dW1 = NN.W1*0; NN.db1 = NN.b1*0;
        if(strcmp(setting.alg,'adadelta'))

            NN.dW5E = NN.W5*0;      NN.db5E = NN.b5*0;
            NN.deltaW5 = NN.W5*0;   NN.deltab5 = NN.b5*0;
            NN.deltaW5E = NN.W5*0;  NN.deltab5E = NN.b5*0;

            NN.dW4E = NN.W4*0;      NN.db4E = NN.b4*0;
            NN.deltaW4 = NN.W4*0;   NN.deltab4 = NN.b4*0;
            NN.deltaW4E = NN.W4*0;  NN.deltab4E = NN.b4*0;

            NN.dW3E = NN.W3*0;      NN.db3E = NN.b3*0;
            NN.deltaW3 = NN.W3*0;   NN.deltab3 = NN.b3*0;
            NN.deltaW3E = NN.W3*0;  NN.deltab3E = NN.b3*0;

            NN.dW2E = NN.W2*0;      NN.db2E = NN.b2*0;
            NN.deltaW2 = NN.W2*0;   NN.deltab2 = NN.b2*0;
            NN.deltaW2E = NN.W2*0;  NN.deltab2E = NN.b2*0;

            NN.dW1E = NN.W1*0;    	NN.db1E = NN.b1*0;
            NN.deltaW1 = NN.W1*0;  	NN.deltab1 = NN.b1*0;
            NN.deltaW1E = NN.W1*0; 	NN.deltab1E = NN.b1*0;
        end
    elseif(strcmp(NN.shape, 'block'))
        NN.W1 = (rand(NN.D0, NN.D1)-0.5)/10;
        NN.b1 = (rand(NN.D1, 1)-0.5)/10;

        for blockID = 1:NN.nBlocks
            NN.W2{blockID} = (rand(NN.D1, NN.D2)-0.5)/10;
            NN.b2{blockID} = (rand(NN.D2,1)-0.5)/10;

            NN.W3{blockID} = (rand(NN.D1, NN.D2^2)-0.5)/10;
            NN.b3{blockID} = (rand(NN.D2^2,1)-0.5)/10;

            NN.W4{blockID} = (rand(NN.D2, NN.D3)-0.5)/10;
        end
        NN.b4 = (rand(NN.D3, 1)-0.5)/10;

        NN.W5 = (rand(NN.D3, NN.D0)-0.5)/10;
        NN.b5 = (rand(NN.D0, 1)-0.5)/10;

        % intermediate variabls in backpropagation
        NN.delta3      = zeros(NN.D0, setting.mbSize*setting.L);
        NN.delta2      = zeros(NN.D3, setting.mbSize*setting.L);
        for blockID = 1:NN.nBlocks
            NN.delta1{blockID}      = zeros(NN.D2, setting.mbSize*setting.L);
            NN.deltaMu{blockID}     = zeros(NN.D2, setting.mbSize);
            NN.deltaAT{blockID}     = zeros(NN.D2, NN.D2, setting.mbSize);
            NN.deltaAM{blockID}     = zeros(NN.D2^2, setting.mbSize);
            NN.deltaBeta{blockID}   = zeros(NN.D2^2, setting.mbSize);
        end
        NN.delta0      = zeros(NN.D1, setting.mbSize);

        % gradient variables used in AdaDelta updating of parameters
        NN.dW5 = NN.W5*0; NN.db5 = NN.b5*0;
        NN.db4 = NN.b4*0;
        for blockID = 1:NN.nBlocks
            NN.dW4{blockID} = NN.W4{blockID}*0; 
            NN.dW3{blockID} = NN.W3{blockID}*0; 
            NN.db3{blockID} = NN.b3{blockID}*0;
            NN.dW2{blockID} = NN.W2{blockID}*0; 
            NN.db2{blockID} = NN.b2{blockID}*0;
        end
        NN.dW1 = NN.W1*0; NN.db1 = NN.b1*0;
        % extra variables if use AdaDelta to optimize the parameters
        if(strcmp(setting.alg,'adadelta'))
            NN.dW5E = NN.W5*0;      NN.db5E = NN.b5*0;
            NN.deltaW5 = NN.W5*0;   NN.deltab5 = NN.b5*0;
            NN.deltaW5E = NN.W5*0;  NN.deltab5E = NN.b5*0;
            
            for blockID = 1:NN.nBlocks
                NN.dW4E{blockID} = NN.W4{blockID}*0;      NN.db4E = NN.b4*0;
                NN.deltaW4{blockID} = NN.W4{blockID}*0;   NN.deltab4 = NN.b4*0;
                NN.deltaW4E{blockID} = NN.W4{blockID}*0;  NN.deltab4E = NN.b4*0;

                NN.dW3E{blockID} = NN.W3{blockID}*0;      
                NN.db3E{blockID} = NN.b3{blockID}*0;
                NN.deltaW3{blockID} = NN.W3{blockID}*0;   
                NN.deltab3{blockID} = NN.b3{blockID}*0;
                NN.deltaW3E{blockID} = NN.W3{blockID}*0;  
                NN.deltab3E{blockID} = NN.b3{blockID}*0;

                NN.dW2E{blockID} = NN.W2{blockID}*0;      
                NN.db2E{blockID} = NN.b2{blockID}*0;
                NN.deltaW2{blockID} = NN.W2{blockID}*0;   
                NN.deltab2{blockID} = NN.b2{blockID}*0;
                NN.deltaW2E{blockID} = NN.W2{blockID}*0;  
                NN.deltab2E{blockID} = NN.b2{blockID}*0;
            end
            NN.dW1E = NN.W1*0;    	NN.db1E = NN.b1*0;
            NN.deltaW1 = NN.W1*0;  	NN.deltab1 = NN.b1*0;
            NN.deltaW1E = NN.W1*0; 	NN.deltab1E = NN.b1*0;
        end
    else
        error('the posterior should be either "diag" or "block"');
    end
end
