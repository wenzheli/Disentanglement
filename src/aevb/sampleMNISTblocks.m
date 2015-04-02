function img = sampleMNISTblocks(nBlocks)

    % part 1. load models and choose the one with maximal LL
    load(['sgdMNIST' num2str(nBlocks) 'blocks.mat']);
    nModels = length(model);
    for i=1:nModels
        l(i) = model{i}.Loss(end);
    end
    [~, mid] = max(l);
    model = model{mid}; 

    % part 2. load data and estimate the marginal posterior distribution
    load data/batchtraindata.mat;
    dataTr = batchdata';
    clear batchdata;
    [ms,vs,M,V] = aevbStat(model, dataTr);

    % part 3. sample images: firstly sample z then sample x|z
    s = 1000;
    rv = mvnrnd(M, V, s);
    for blockID = 1:nBlocks
        z{blockID} = rv(:,(blockID-1)*2+1:blockID*2);
        for i=1:s
            p{blockID}(i) = mvncdf(z{blockID}(i,:),ms{blockID}(:)',vs{blockID}(1:2,1:2));
        end
        z{blockID} = [z{blockID} p{blockID}(:)];
        z{blockID} = sortrows(z{blockID},3);
        z{blockID} = z{blockID}(1:40:end,1:2); % select 25 samples
        N(blockID) = length(z{blockID});
    end

    
    %% start sampling
    W5 = model.W5;
    W4 = model.W4;

    b5 = model.b5;
    b4 = model.b4;

    img = zeros(29*N(1)-1,29*N(2)-1);
    id = 1;
    for i=1:N(1)
        for j=1:N(2)
            Z1 = [z{1}(i,1); z{1}(i,2)];
            Z2 = [z{2}(j,2); z{2}(j,2)];
            h2 = tanh(W4{1}'*Z1 + b4{1} + W4{2}'*Z2 + b4{2});
            x = 1./(1+exp(-bsxfun(@plus, W5'*h2, b5))); 
            x = reshape(x,[28,28]);
            img((i-1)*29+1:i*29-1,(j-1)*29+1:j*29-1) = x';
            id=id+1;
        end
    end
end% the struction of the model
% q(z|x): dim-784 Xs --> dim-200 hu --> dim-2 Gaussian Z
% p(x|z): dim-2 Gaussian Z --> dim-200 hu --> dim-784 Xs

function [ms,vs,M,V] = aevbStat(model, dataTr)
    
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
    
    if(iscell(W2))
        nBlocks = length(W2);
    else
        nBlocks = 1;
    end

    %% part 0. feed-forward propagation, encoding and then decoding for reconstruction
    N = 10000;
    firstIdx = 1;
    lastIdx = N;

    X1 = dataTr(:, firstIdx:lastIdx);
    h1 = tanh(bsxfun(@plus, W1'*X1, b1));
    for blockID = 1:nBlocks
        mu{blockID} = bsxfun(@plus, W2{blockID}'*h1, b2{blockID});
        beta{blockID} = bsxfun(@plus, W3{blockID}'*h1, b3{blockID});
        Z{blockID} = mu{blockID};
    end

    blockID = 1;
    h2 = (bsxfun(@plus, W4{blockID}'*Z{blockID}, b4{blockID}));
    for blockID=2:nBlocks
        h2 = h2+(bsxfun(@plus, W4{blockID}'*Z{blockID}, b4{blockID}));
    end
    h2 = tanh(h2);

    Y = 1./(1+exp(-bsxfun(@plus, W5'*h2, b5))); 

    Zs = cell(10,nBlocks);
    ms = cell(1,nBlocks);
    vs = cell(1,nBlocks);
    for blockID = 1:nBlocks
        ms{blockID} = mean(Z{blockID},2);
        vs{blockID} = cov(Z{blockID}');
    end

    z = Z{1}';
    for blockID = 2:nBlocks
        z = [z Z{blockID}'];
    end
    % full mean and covariance matrix
    M = mean(z);
    V = cov(z);

end