function valid = verifyValues(NN)

    valid = 1;
    
    %% check if W variables are valid
    valid = valid*checkValidNumber(NN.W5);

    valid = valid*checkValidNumber(NN.b1);
    
    valid = valid*checkValidNumber(NN.h2);
    
    if(isfield(NN,'loss'))
        valid = valid*checkValidNumber(NN.loss);
    end
end

function bool = checkValidNumber(X)
    bool = 1;
    if(isnan(sum(X(:))) || isinf(sum(X(:))))
        bool=0;
    end
end
