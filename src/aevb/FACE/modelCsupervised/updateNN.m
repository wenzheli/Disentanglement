function NN = updateNN(NN, setting)
    if(strcmp(NN.shape, 'diag'))
        if(strcmp(setting.alg, 'sgd'))
            % SGD update parameters
            NN.W5 = NN.W5 + NN.lrate.*NN.dW5; NN.b5 = NN.b5 + NN.lrate.*NN.db5;
            NN.W4 = NN.W4 + NN.lrate.*NN.dW4; NN.b4 = NN.b4 + NN.lrate.*NN.db4;
            NN.W3 = NN.W3 + NN.lrate.*NN.dW3; NN.b3 = NN.b3 + NN.lrate.*NN.db3;
            NN.W2 = NN.W2 + NN.lrate.*NN.dW2; NN.b2 = NN.b2 + NN.lrate.*NN.db2;
            NN.W1 = NN.W1 + NN.lrate.*NN.dW1; NN.b1 = NN.b1 + NN.lrate.*NN.db1;
            for classID = 1:NN.nClasses
                NN.Wc{classID} = NN.Wc{classID} + NN.lrate{classID}.*NN.dWc{classID}; 
                NN.bc{classID} = NN.bc{classID} + NN.lrate{classID}.*NN.dbc{classID};
            end
        elseif(strcmp(setting.alg, 'adadelta'))
            %% ADA update of parameters and statistics
            NN.dW5E = NN.rho*NN.dW5E + (1-NN.rho)*NN.dW5.^2;
            NN.db5E = NN.rho*NN.db5E + (1-NN.rho)*NN.db5.^2;
            NN.dW1E = NN.rho*NN.dW1E + (1-NN.rho)*NN.dW1.^2;
            NN.db1E = NN.rho*NN.db1E + (1-NN.rho)*NN.db1.^2;
            
            lrateW5 = sqrt(NN.deltaW5E+NN.const)./sqrt(NN.dW5E+NN.const);
            lrateW1 = sqrt(NN.deltaW1E+NN.const)./sqrt(NN.dW1E+NN.const);
            lrateb5 = sqrt(NN.deltab5E+NN.const)./sqrt(NN.db5E+NN.const);
            lrateb1 = sqrt(NN.deltab1E+NN.const)./sqrt(NN.db1E+NN.const);

            NN.deltaW5 = lrateW5.*NN.dW5; 
            NN.deltab5 = lrateb5.*NN.db5;
            NN.deltaW1 = lrateW1.*NN.dW1; 
            NN.deltab1 = lrateb1.*NN.db1;

            NN.deltaW5E = NN.rho*NN.deltaW5E + (1-NN.rho)*NN.deltaW5.^2;
            NN.deltaW1E = NN.rho*NN.deltaW1E + (1-NN.rho)*NN.deltaW1.^2;
            NN.deltab5E = NN.rho*NN.deltab5E + (1-NN.rho)*NN.deltab5.^2;
            NN.deltab1E = NN.rho*NN.deltab1E + (1-NN.rho)*NN.deltab1.^2;

            NN.db4E = NN.rho*NN.db4E + (1-NN.rho)*NN.db4.^2;
            lrateb4 = sqrt(NN.deltab4E+NN.const)./sqrt(NN.db4E+NN.const);
            NN.deltab4 = lrateb4.*NN.db4;
            NN.deltab4E= NN.rho*NN.deltab4E + (1-NN.rho)*NN.deltab4.^2;

            NN.dW4E = NN.rho*NN.dW4E + (1-NN.rho)*NN.dW4.^2;   
            NN.dW3E = NN.rho*NN.dW3E + (1-NN.rho)*NN.dW3.^2;   
            NN.db3E = NN.rho*NN.db3E + (1-NN.rho)*NN.db3.^2;
            NN.dW2E = NN.rho*NN.dW2E + (1-NN.rho)*NN.dW2.^2;   
            NN.db2E = NN.rho*NN.db2E + (1-NN.rho)*NN.db2.^2;

            lrateW4 = sqrt(NN.deltaW4E+NN.const)./sqrt(NN.dW4E+NN.const);
            lrateW3 = sqrt(NN.deltaW3E+NN.const)./sqrt(NN.dW3E+NN.const);
            lrateW2 = sqrt(NN.deltaW2E+NN.const)./sqrt(NN.dW2E+NN.const);

            lrateb3 = sqrt(NN.deltab3E+NN.const)./sqrt(NN.db3E+NN.const);
            lrateb2 = sqrt(NN.deltab2E+NN.const)./sqrt(NN.db2E+NN.const);

            NN.deltaW4 = lrateW4.*NN.dW4; 
            NN.deltaW3 = lrateW3.*NN.dW3;
            NN.deltab3 = lrateb3.*NN.db3;
            NN.deltaW2 = lrateW2.*NN.dW2; 
            NN.deltab2 = lrateb2.*NN.db2;

            NN.deltaW4E = NN.rho*NN.deltaW4E + (1-NN.rho)*NN.deltaW4.^2;
            NN.deltaW3E = NN.rho*NN.deltaW3E + (1-NN.rho)*NN.deltaW3.^2;
            NN.deltaW2E = NN.rho*NN.deltaW2E + (1-NN.rho)*NN.deltaW2.^2;

            NN.deltab3E = NN.rho*NN.deltab3E + (1-NN.rho)*NN.deltab3.^2;
            NN.deltab2E = NN.rho*NN.deltab2E + (1-NN.rho)*NN.deltab2.^2;

            NN.W5 = NN.W5+NN.deltaW5;    NN.b5 = NN.b5+NN.deltab5;
            NN.W4 = NN.W4+NN.deltaW4;    NN.b4 = NN.b4+NN.deltab4;
            NN.W3 = NN.W3+NN.deltaW3;    NN.b3 = NN.b3+NN.deltab3;
            NN.W2 = NN.W2+NN.deltaW2;    NN.b2 = NN.b2+NN.deltab2;
            NN.W1 = NN.W1+NN.deltaW1;    NN.b1 = NN.b1+NN.deltab1;
           
            for classID =1 :NN.nBlocks
                NN.dWcE{classID} = NN.rho*NN.dWcE{classID} + (1-NN.rho)*NN.dWc{classID}.^2;
                NN.dbcE{classID} = NN.rho*NN.dbcE{classID} + (1-NN.rho)*NN.dbc{classID}.^2;
                NN.lrateWc{classID} = sqrt(NN.deltaWcE{classID}+NN.const)./sqrt(NN.dWcE{classID}+NN.const);
                NN.lratebc{classID} = sqrt(NN.deltabcE{classID}+NN.const)./sqrt(NN.dbcE{classID}+NN.const);
                NN.deltaWc{classID} = NN.lrateWc{classID}.*NN.dWc{classID}; 
                NN.deltabc{classID} = NN.lratebc{classID}.*NN.dbc{classID};
                NN.deltaWcE{classID} = NN.rho*NN.deltaWcE{classID} + (1-NN.rho)*NN.deltaWc{classID}.^2;
                NN.deltabcE{classID} = NN.rho*NN.deltabcE{classID} + (1-NN.rho)*NN.deltabc{classID}.^2;
                NN.Wc{classID} = NN.Wc{classID} + NN.deltaWc{classID};    NN.bc{classID} = NN.bc{classID}+NN.deltabc{classID};
            end
        elseif(strcmp(setting.alg, 'adam'))
        end
    elseif(strcmp(NN.shape,'block'))
        if(strcmp(setting.alg, 'sgd'))
            % SGD update parameters
            NN.W5 = NN.W5 + NN.lrate.*NN.dW5; NN.b5 = NN.b5 + NN.lrate.*NN.db5;
            NN.b4 = NN.b4 + NN.lrate.*NN.db4;
            for blockID = 1:NN.nBlocks
                NN.W4{blockID} = NN.W4{blockID} + NN.lrate.*NN.dW4{blockID}; 
                NN.W3{blockID} = NN.W3{blockID} + NN.lrate.*NN.dW3{blockID}; 
                NN.b3{blockID} = NN.b3{blockID} + NN.lrate.*NN.db3{blockID};
                NN.W2{blockID} = NN.W2{blockID} + NN.lrate.*NN.dW2{blockID}; 
                NN.b2{blockID} = NN.b2{blockID} + NN.lrate.*NN.db2{blockID};
            end
            NN.W1 = NN.W1 + NN.lrate.*NN.dW1; NN.b1 = NN.b1 + NN.lrate.*NN.db1;
            for classID = 1:NN.nClasses
                NN.Wc{classID} = NN.Wc{classID} + NN.lrate*NN.dWc{classID}; 
                NN.bc{classID} = NN.bc{classID} + NN.lrate*NN.dbc{classID};
            end
        elseif(strcmp(setting.alg, 'adadelta'))
            %% ADA update of parameters and statistics
            NN.dW5E = NN.rho*NN.dW5E + (1-NN.rho)*NN.dW5.^2;
            NN.db5E = NN.rho*NN.db5E + (1-NN.rho)*NN.db5.^2;
            NN.db4E = NN.rho*NN.db4E + (1-NN.rho)*NN.db4.^2;
            NN.dW1E = NN.rho*NN.dW1E + (1-NN.rho)*NN.dW1.^2;
            NN.db1E = NN.rho*NN.db1E + (1-NN.rho)*NN.db1.^2;

            lrateW5 = sqrt(NN.deltaW5E+NN.const)./sqrt(NN.dW5E+NN.const);
            lrateb5 = sqrt(NN.deltab5E+NN.const)./sqrt(NN.db5E+NN.const);
            lrateb4 = sqrt(NN.deltab4E+NN.const)./sqrt(NN.db4E+NN.const);
            lrateW1 = sqrt(NN.deltaW1E+NN.const)./sqrt(NN.dW1E+NN.const);
            lrateb1 = sqrt(NN.deltab1E+NN.const)./sqrt(NN.db1E+NN.const);

            NN.deltaW5 = lrateW5.*NN.dW5; 
            NN.deltab5 = lrateb5.*NN.db5;
            NN.deltab4 = lrateb4.*NN.db4;
            NN.deltaW1 = lrateW1.*NN.dW1; 
            NN.deltab1 = lrateb1.*NN.db1;

            NN.deltaW5E = NN.rho*NN.deltaW5E + (1-NN.rho)*NN.deltaW5.^2;
            NN.deltab5E = NN.rho*NN.deltab5E + (1-NN.rho)*NN.deltab5.^2;
            NN.deltab4E= NN.rho*NN.deltab4E + (1-NN.rho)*NN.deltab4.^2;
            NN.deltaW1E = NN.rho*NN.deltaW1E + (1-NN.rho)*NN.deltaW1.^2;
            NN.deltab1E = NN.rho*NN.deltab1E + (1-NN.rho)*NN.deltab1.^2;
            
            for blockID = 1:NN.nBlocks
                NN.dW4E{blockID} = NN.rho*NN.dW4E{blockID} + (1-NN.rho)*NN.dW4{blockID}.^2;   
                NN.dW3E{blockID} = NN.rho*NN.dW3E{blockID} + (1-NN.rho)*NN.dW3{blockID}.^2;   
                NN.db3E{blockID} = NN.rho*NN.db3E{blockID} + (1-NN.rho)*NN.db3{blockID}.^2;
                NN.dW2E{blockID} = NN.rho*NN.dW2E{blockID} + (1-NN.rho)*NN.dW2{blockID}.^2;   
                NN.db2E{blockID} = NN.rho*NN.db2E{blockID} + (1-NN.rho)*NN.db2{blockID}.^2;

                lrateW4 = sqrt(NN.deltaW4E{blockID}+NN.const)./sqrt(NN.dW4E{blockID}+NN.const);
                lrateW3 = sqrt(NN.deltaW3E{blockID}+NN.const)./sqrt(NN.dW3E{blockID}+NN.const);
                lrateW2 = sqrt(NN.deltaW2E{blockID}+NN.const)./sqrt(NN.dW2E{blockID}+NN.const);
                lrateb3 = sqrt(NN.deltab3E{blockID}+NN.const)./sqrt(NN.db3E{blockID}+NN.const);
                lrateb2 = sqrt(NN.deltab2E{blockID}+NN.const)./sqrt(NN.db2E{blockID}+NN.const);

                NN.deltaW4{blockID} = lrateW4.*NN.dW4{blockID}; 
                NN.deltaW3{blockID} = lrateW3.*NN.dW3{blockID};
                NN.deltab3{blockID} = lrateb3.*NN.db3{blockID};
                NN.deltaW2{blockID} = lrateW2.*NN.dW2{blockID}; 
                NN.deltab2{blockID} = lrateb2.*NN.db2{blockID};

                NN.deltaW4E{blockID} = NN.rho*NN.deltaW4E{blockID} + (1-NN.rho)*NN.deltaW4{blockID}.^2;
                NN.deltaW3E{blockID} = NN.rho*NN.deltaW3E{blockID} + (1-NN.rho)*NN.deltaW3{blockID}.^2;
                NN.deltab3E{blockID} = NN.rho*NN.deltab3E{blockID} + (1-NN.rho)*NN.deltab3{blockID}.^2;
                NN.deltaW2E{blockID} = NN.rho*NN.deltaW2E{blockID} + (1-NN.rho)*NN.deltaW2{blockID}.^2;
                NN.deltab2E{blockID} = NN.rho*NN.deltab2E{blockID} + (1-NN.rho)*NN.deltab2{blockID}.^2;
                
                NN.W4{blockID} = NN.W4{blockID}+NN.deltaW4{blockID};    
                NN.W3{blockID} = NN.W3{blockID}+NN.deltaW3{blockID};    
                NN.b3{blockID} = NN.b3{blockID}+NN.deltab3{blockID};
                NN.W2{blockID} = NN.W2{blockID}+NN.deltaW2{blockID};    
                NN.b2{blockID} = NN.b2{blockID}+NN.deltab2{blockID};
            end

            NN.W5 = NN.W5+NN.deltaW5;    NN.b5 = NN.b5+NN.deltab5;
            NN.b4 = NN.b4+NN.deltab4;
            NN.W1 = NN.W1+NN.deltaW1;    NN.b1 = NN.b1+NN.deltab1;
            for classID = 1:NN.nClasses
                NN.dWcE{classID} = NN.rho*NN.dWcE{classID} + (1-NN.rho)*NN.dWc{classID}.^2;
                NN.dbcE{classID} = NN.rho*NN.dbcE{classID} + (1-NN.rho)*NN.dbc{classID}.^2;
                
                NN.lrateWc{classID} = sqrt(NN.deltaWcE{classID}+NN.const)./sqrt(NN.dWcE{classID}+NN.const);
                NN.lratebc{classID} = sqrt(NN.deltabcE{classID}+NN.const)./sqrt(NN.dbcE{classID}+NN.const);
                
                NN.deltaWc{classID} = NN.lrateWc{classID}.*NN.dWc{classID}; 
                NN.deltabc{classID} = NN.lratebc{classID}.*NN.dbc{classID};
                
                NN.deltaWcE{classID} = NN.rho*NN.deltaWcE{classID} + (1-NN.rho)*NN.deltaWc{classID}.^2;
                NN.deltabcE{classID} = NN.rho*NN.deltabcE{classID} + (1-NN.rho)*NN.deltabc{classID}.^2;
                
                NN.Wc{classID} = NN.Wc{classID}+NN.deltaWc{classID};    
                NN.bc{classID} = NN.bc{classID}+NN.deltabc{classID};
            end
        elseif(strcmp(setting.alg, 'adam'))
        end
    else
        error('posterior must be either "block" or "diag"');
    end
end
