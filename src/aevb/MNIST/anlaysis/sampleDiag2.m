function img = sampleDiag2(name)
    load(['../results/' name '.mat']);
    
    nModels = size(model,1)*size(model,2);
    if(nModels>1)
        for i=1:nModels
            l(i) = model{i}.LL(end,1);
            if(l(i)==0)
                l(i) = -10000;
            end
        end
        [~, mid] = max(l);
        model = model{mid}; 
    else
        model = model(1);
    end
    
    if(length(model.b2)~=2)
        error('number of hidden units must be 2');
    end
    % part 2. load data and estimate the marginal posterior distribution
    load ../../data/batchtraindata.mat;
    dataTr = batchdata';
    clear batchdata;
    [Mdata,Vdata,Vtrue] = aevbStat(model, dataTr);
    
    % part 3. sample images: firstly sample z then sample x|z
    % sample each blocks respectively
    s = 6000;
    rv = mvnrnd(Mdata, Vdata, s);
    rv = sort(rv);
    
    z = rv(100:200:end,:);
    N = size(z,1);

    
    %% start sampling
    img = zeros(29*N-1,29*N-1);
    for i=1:N
        for j=1:N
            Z = [z(i,1); z(j,2)];
            h2 = tanh(model.W4'*Z + model.b4);
            x = 1./(1+exp(-bsxfun(@plus, model.W5'*h2, model.b5)));
            x = reshape(x,[28,28]);
            img((i-1)*29+1:i*29-1,(j-1)*29+1:j*29-1) = x';
        end
    end

    figure, imshow(img, []);
    fprintf(2,'mean\n');
    disp(Mdata);
    fprintf(2,'cov_data\n');
    disp(Vdata);
    fprintf(2,'cov_post\n');
    disp(Vtrue);
end

function [Mdata, Vdata, Vtrue] = aevbStat(model, dataTr)
    
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

    D2 = length(b2);

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
    mu = bsxfun(@plus, W2'*h1, b2);
    beta = bsxfun(@plus, W3'*h1, b3);
    beta = exp(beta);
    
    Vtrue = diag(mean(beta,2));
    Mdata = mean(mu');
    Vdata = cov(mu');
end
