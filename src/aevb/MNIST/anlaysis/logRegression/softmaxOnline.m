function w0 = softmaxOnline(train, devel, nc, lr, scaler, lambda)
% train, test data
%   each containing feature and label matrix
%   (N_example x F_dim) and(N_example x N_class)
%   nc:  number of class
%   lr:  learning rate
%   scaler: learning rate decay
%   lambda: L2 weight decay

    dim = size(train.X,2);
    % initialization 
    w = rand(dim*(nc-1),1)-0.5;
    
    xTr = train.X;
    yTr = train.Y;
    N = size(xTr,1);
    if(size(yTr,2)~=nc)
        keyboard
        error('classes mismatch');
    end
    if(size(yTr,1)~=N)
        keyboard
        error('samples mismatch');
    end

    err = 1; 
    maxEpoch = 25;
    batchsize = 50;
    numbatch = ceil(N/batchsize);
    for epoch = 1:maxEpoch
        
        for batch=0:numbatch-1
            idx1 = batch*batchsize+1;
            idx2 = min((batch+1)*batchsize, N);
%             if(batch==3071)
%                 keyboard
%             end
            [~,gradient]=softmax(w,xTr(idx1:idx2,:),yTr(idx1:idx2,:),lambda);
%             if(rem(batch,500)==1)
% %                 batch
%                 mean(gradient)
%             end
            if(isnan(sum(gradient)))
                keyboard
            end
            w = w-lr*gradient;
            if(rem(batch,50)==0)
                [~,acc] = softmaxPredict(w,devel.X, devel.Y);
                if((1-acc)<err)
                    err = 1-acc;
                    w0 = w;
                end
            end
        end
        
        [~,acc] = softmaxPredict(w,devel.X, devel.Y);

%         fprintf('epoch %d, accuracy %f\n', epoch, acc);

        if((1-acc)<err)
            err = 1-acc;
            w0 = w;
        end
        
        lr = lr*scaler;
    end
end
