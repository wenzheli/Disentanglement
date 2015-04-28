function [img1, img2, img3] = sampleBlock(name)
    load(['../results/' name '.mat']);
    nBlocks = 2;
    D2 = 2;
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
    
    % part 2. load data and estimate the marginal posterior distribution
    load ../../data/batchtraindata.mat;
    dataTr = batchdata';
    clear batchdata;
    [ms,vs,Mdata,Vdata,Vtrue] = aevbStat(model, dataTr);

    % part 3. sample images: firstly sample z then sample x|z
    % sample each blocks respectively
    s = 6000;

    rv = mvnrnd(Mdata, Vdata, s);
    rv = sort(rv);
    borderLow = min(rv);
    borderUp = max(rv);
    borderDist = borderUp-borderLow;
    borderLow = borderLow+borderDist/20;
    borderUp = borderUp - borderDist/20;
    borderUnit = (borderUp-borderLow)/25;
   
    Coords = bsxfun(@plus, bsxfun(@times, repmat((0.5:1:25)',[1,4]), borderUnit), borderLow);
    

    %% start sampling
    W5 = model.W5;
    W4 = model.W4;

    b5 = model.b5;
    b4 = model.b4;

    N = zeros(4,1)+25;

    
    img1 = cell(25,1); % block 1 fixed
    id = 1;
    imgLabel1 = zeros(1,2);
    for i=2:5:N(1)
        for j=2:5:N(1)
            img1{id} = zeros(29*N(2)-1,29*N(2)-1);
            for k=1:N(2)
                for l=1:N(2)
                    Z1 = [Coords(i,1); Coords(j,2)]+(rand(2,1)-0.5).*borderUnit(1:2)';
                    Z2 = [Coords(k,3); Coords(l,4)];
                    h2 = tanh(W4{1}'*Z1 + W4{2}'*Z2 + b4);
                    x = 1./(1+exp(-bsxfun(@plus, W5'*h2, b5)));
                    x = reshape(x,[28,28]);
                    img1{id}((k-1)*29+1:k*29-1,(l-1)*29+1:l*29-1) = x';
                end
            end
            imgLabel1(id,:) = Z1';
            id=id+1;
        end
    end
   
    for i=1:length(img1)
        figure, imshow(img1{i},[]);
        title(['Z(1,2): (' num2str(imgLabel1(i,1)) ', ' num2str(imgLabel1(i,2)) ')'], 'FontSize',12,'FontWeight','Demi'); 
    end

    img2 = cell(25,1); % block 1 fixed
    id = 1;
    imgLabel2 = zeros(1,2);
    for i=2:5:N(2)
        for j=2:5:N(2)
            img2{id} = zeros(29*N(1)-1,29*N(1)-1);
            for k=1:N(1)
                for l=1:N(1)
                    Z2 = [Coords(i,3); Coords(j,4)]+ (rand(2,1)-0.5).*borderUnit(3:4)';
                    Z1 = [Coords(k,1); Coords(l,2)];
                    h2 = tanh(W4{1}'*Z1 + W4{2}'*Z2 + b4);
                    x = 1./(1+exp(-bsxfun(@plus, W5'*h2, b5)));
                    x = reshape(x,[28,28]);
                    img2{id}((k-1)*29+1:k*29-1,(l-1)*29+1:l*29-1) = x';
                end
            end
            imgLabel2(id,:) = Z2';
            id=id+1;
        end
    end
    for i=1:length(img2)
        figure, imshow(img2{i},[]);
        title(['Z(3,4): (' num2str(imgLabel2(i,1)) ', ' num2str(imgLabel2(i,2)) ')'], 'FontSize',12,'FontWeight','Demi'); 
    end

    img3 = cell(2,1);
    for k=1:N(2)
        for l=1:N(2)
            Z1 = [Mdata(1); Mdata(2)];
            Z2 = [Coords(k,3); Coords(l,4)];
            h2 = tanh(W4{1}'*Z1 + W4{2}'*Z2 + b4);
            x = 1./(1+exp(-bsxfun(@plus, W5'*h2, b5)));
            x = reshape(x,[28,28]);
            img3{1}((k-1)*29+1:k*29-1,(l-1)*29+1:l*29-1) = x';
        end
    end
    for k=1:N(1)
        for l=1:N(1)
            Z2 = [Mdata(3); Mdata(4)];
            Z1 = [Coords(k,1); Coords(l,2)];
            h2 = tanh(W4{1}'*Z1 + W4{2}'*Z2 + b4);
            x = 1./(1+exp(-bsxfun(@plus, W5'*h2, b5)));
            x = reshape(x,[28,28]);
            img3{2}((k-1)*29+1:k*29-1,(l-1)*29+1:l*29-1) = x';
        end
    end
    fprintf(2,'Mean:\n');
    disp(Mdata);
    fprintf(2,'data coVariance\n');
    disp(Vdata);
    fprintf(2,'posterior Covariance, average\n');
    disp(Vtrue);
    
end

function [ms, vs, Mdata, Vdata, Vtrue] = aevbStat(model, dataTr)
    
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
    
    D2 = length(b2{1});

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
        beta{blockID} = exp(beta{blockID});
        at{blockID} = reshape(beta{blockID}, [D2, D2, N]);
        Z{blockID} = mu{blockID};
    end
    for i=1:N
        at{blockID}(:,:,i) = at{blockID}(:,:,i)*at{blockID}(:,:,i)';
    end
    Vtrue = zeros(D2*nBlocks);
    for blockID = 1:nBlocks
        Vtrue((blockID-1)*D2+1:blockID*D2, (blockID-1)*D2+1:blockID*2) = mean(at{blockID},3);
    end
    
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
    Mdata = mean(z);
    Vdata = cov(z);
end
