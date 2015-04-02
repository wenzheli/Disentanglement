% the struction of the model
% q(z|x): dim-784 Xs --> dim-200 hu --> dim-2 Gaussian Z
% p(x|z): dim-2 Gaussian Z --> dim-200 hu --> dim-784 Xs

load data/batchtraindata.mat;
dataTr = batchdata';
clear batchdata;

load sgdMNIST2blocks.mat
model = model{2};
nBlocks = 2;

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

%% part 1. select 100 samples to compare the original and reconstructed images
img0 = zeros(29*10-1,29*10-1);
img1 = zeros(29*10-1,29*10-1);
id = 1;
for i=1:10
    for j=1:10
        x = X1(:,id);
        x = reshape(x,[28,28]);
        img1((i-1)*29+1:i*29-1,(j-1)*29+1:j*29-1) = x;

        x = Y(:,id);
        x = reshape(x,[28,28]);
        img1((i-1)*29+1:i*29-1,(j-1)*29+1:j*29-1) = x;
        id=id+1;
    end
end
figure, subplot(1,2,1), imshow(img0,[]);
title('original', 'FontSize',12','FontWeigth','Demi');
subplot(1,2,2), imshow(img1,[]);
title('reconstruct', 'FontSize',12','FontWeigth','Demi');

Zs = cell(10,nBlocks);
m = cell(1,nBlocks);
v = cell(1,nBlocks);
for blockID = 1:nBlocks
    for i=1:10
        Zs{i, blockID} = Z{1}(:,batchlabel(1:10000)==i);
    end
    m{blockID} = mean(Z{blockID},2);
    v{blockID} = cov(Z{blockID}');
end

z = Z{1}';
for blockID = 2:nBlocks
    z = [z Z{blockID}'];
end
% full mean and covariance matrix
M = mean(z);
V = cov(z);

% 
figure, 
for blockID = 1:nBlocks
    subplot(1,2,blockID), plot(Zs1{1,blockID}(1,:), Zs1{1, blockID}(2,:),'*');
    hold on, plot(Zs1{2 ,blockID}(1,:), Zs1{2 ,blockID}(2,:),'r*')
    hold on, plot(Zs1{3 ,blockID}(1,:), Zs1{3 ,blockID}(2,:),'g*')
    hold on, plot(Zs1{4 ,blockID}(1,:), Zs1{4 ,blockID}(2,:),'k*')
    hold on, plot(Zs1{5 ,blockID}(1,:), Zs1{5 ,blockID}(2,:),'c*')
    hold on, plot(Zs1{6 ,blockID}(1,:), Zs1{6 ,blockID}(2,:),'m*')
    hold on, plot(Zs1{7 ,blockID}(1,:), Zs1{7 ,blockID}(2,:),'y*')
    hold on, plot(Zs1{8 ,blockID}(1,:), Zs1{8 ,blockID}(2,:),'gd')
    hold on, plot(Zs1{9 ,blockID}(1,:), Zs1{9 ,blockID}(2,:),'rd')
    hold on, plot(Zs1{10 ,blockID}(1,:), Zs1{10 ,blockID}(2,:),'kd')
end
