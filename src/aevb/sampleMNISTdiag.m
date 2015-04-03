function sampleMNISTdiag()

    load modelMNISTdiag_prior.mat
    nModel = length(model);
    for i=1:nModel
        if(isfield(model{i},'LL'))
            l(i) = model{i}.LL(end);
        else
            l(i) = model{i}.Loss(end);
        end
    end
    [~, mid] = max(l);
    model = model{mid};  %select the model with the smallest reconstruction error

    
    W5 = model.W5;
    W4 = model.W4;
    W3 = model.W3;
    W2 = model.W2;
    W1 = model.W1;

    b5 = model.b5;
    b4 = model.b4;
    b3 = model.b3;
    b2 = model.b2;
    b1 = model.b1;

    % it is easy to sample images when there are only 2 nodes in the hidden space
    % the only task is to sample z1 and z2

    [mu, sigma] = modelStat(model);
    
    cdfP = 0.1:0.025:0.9;
    z1 = norminv(cdfP, mu(1), sigma(1,1));
    z2 = norminv(cdfP, mu(2), sigma(2,2));
    
    % heuristic sampling:
%     z1 = -10:3:20;
%     z2 = -10:3:20;
%     z1 = [z1 -3.5:0.75:11]; % finer-grained sampling around the mean
%     z2 = [z2 -3.5:0.75:11]; 

    % z1 = -5:0.5:5;
    % z2 = -5:0.5:5;
    % z1 = [z1 -2:0.2:2];
    % z2 = [z2 -2:0.2:2];

    z1 = sort(unique(z1));
    z2 = sort(unique(z2));
    N1 = length(z1);
    N2 = length(z2);

    % there could be other sampling method, e.g. sampling z ~ E_x[q(z|x)]

    img = zeros(29*N1-1,29*N2-1);
    id = 1;
    for i=1:N1
        for j=1:N2
            Z = [z1(i); z2(j)];
            h2 = tanh(bsxfun(@plus, W4'*Z, b4));
            x = 1./(1+exp(-bsxfun(@plus, W5'*h2, b5))); 
            x = reshape(x,[28,28]);
            img((i-1)*29+1:i*29-1,(j-1)*29+1:j*29-1) = x';
            id=id+1;
        end
    end

    figure, imshow(img, []);
    keyboard
end

function [m, v] = modelStat(model)

    load ../data/batchtraindata.mat;
    dataTr = batchdata';
    clear batchdata;

    W5 = model.W5;
    W4 = model.W4;
    W3 = model.W3;
    W2 = model.W2;
    W1 = model.W1;

    b5 = model.b5;
    b4 = model.b4;
    b3 = model.b3;
    b2 = model.b2;
    b1 = model.b1;

    %% part 0. feed-forward propagation, encoding and then decoding for reconstruction
    N = 10000;
    firstIdx = 1;
    lastIdx = N;

    X1 = dataTr(:, firstIdx:lastIdx);
    h1 = tanh(bsxfun(@plus, W1'*X1, b1));
    mu = bsxfun(@plus, W2'*h1, b2);
    m = mean(mu');
    v = cov(mu');
end