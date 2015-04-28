function [M, Vtrue, Vdata, Z] = ffDiag(name)
% a quick evaluation of the model
% 1. reconstruction capability
% 2. latent space visualization
% 3. statistics of latent space

%% part 0: load model and preprocess to extract basic model information
load(['../results/' name '.mat']);

nModel = length(model);
if(nModel>1)
    for mid=1:nModel
        l(mid) = sum(model{mid}.LL(end,:));
    end
    [~, mid] = max(l);
    model = model{mid};
else
    model = model(1);
end

W5 = model.W5;  b5 = model.b5;
W4 = model.W4;  b4 = model.b4;
W3 = model.W3;  b3 = model.b3;
W2 = model.W2;  b2 = model.b2;
W1 = model.W1;  b1 = model.b1;


%% part 1. feed-forward propagation, encoding and then decoding for reconstruction
load ../../data/batchtraindata.mat;
dataTr = batchdata';
clear batchdata;

N = size(dataTr,2);

X1 = dataTr;
h1 = tanh(bsxfun(@plus, W1'*X1, b1));
mu = bsxfun(@plus, W2'*h1, b2);
beta = bsxfun(@plus, W3'*h1, b3);
lambda = exp(0.5*beta);
Z = mu;

h2 = (bsxfun(@plus, W4'*Z, b4));
h2 = h2+(bsxfun(@plus, W4'*Z, b4));
h2 = tanh(h2);

Y = 1./(1+exp(-bsxfun(@plus, W5'*h2, b5))); 

M = mean(Z,2);
Vdata = cov(Z');
Vtrue = mean(lambda,2);

%% part 2. select 100 samples to compare the original and reconstructed images
img0 = zeros(29*10-1,29*10-1);
img1 = zeros(29*10-1,29*10-1);
id = 1;
for i=1:10
    for j=1:10
        x = X1(:,id);
        x = reshape(x,[28,28]);
        img0((i-1)*29+1:i*29-1,(j-1)*29+1:j*29-1) = x';

        x = Y(:,id);
        x = reshape(x,[28,28]);
        img1((i-1)*29+1:i*29-1,(j-1)*29+1:j*29-1) = x';
        id=id+1;
    end
end

figure, subplot(1,2,1), imshow(img0,[]);
title('original', 'FontSize',12','FontWeight','Demi');
subplot(1,2,2), imshow(img1,[]);
title('reconstruct', 'FontSize',12','FontWeight','Demi');

for i=1:10
    Zs{i} = Z(:,batchlabel(1:10000)==i);
end

% 
D2 = length(model.b2);
if(D2==2)
    H2 = figure;
    set(H2, 'Position',[100,100,500,400]);
    plot(Zs{1}(1,:), Zs{1}(2,:),'*');
    hold on, plot(Zs{2}(1,:), Zs{2}(2,:),'r*')
    hold on, plot(Zs{3}(1,:), Zs{3}(2,:),'g*')
    hold on, plot(Zs{4}(1,:), Zs{4}(2,:),'k*')
    hold on, plot(Zs{5}(1,:), Zs{5}(2,:),'c*')
    hold on, plot(Zs{6}(1,:), Zs{6}(2,:),'m*')
    hold on, plot(Zs{7}(1,:), Zs{7}(2,:),'y*')
    hold on, plot(Zs{8}(1,:), Zs{8}(2,:),'gd')
    hold on, plot(Zs{9}(1,:), Zs{9}(2,:),'rd')
    hold on, plot(Zs{10}(1,:), Zs{10}(2,:),'kd')
    title('2D coding', 'FontSize', 12, 'FontWeight', 'Demi');
end

if(D2==4)
    H2 = figure;
    set(H2, 'Position',[100,100,1000,400]);
    subplot(1,2,1), plot(Zs{1}(1,:), Zs{1}(2,:),'*');
    hold on, plot(Zs{2}(1,:), Zs{2}(2,:),'r*')
    hold on, plot(Zs{3}(1,:), Zs{3}(2,:),'g*')
    hold on, plot(Zs{4}(1,:), Zs{4}(2,:),'k*')
    hold on, plot(Zs{5}(1,:), Zs{5}(2,:),'c*')
    hold on, plot(Zs{6}(1,:), Zs{6}(2,:),'m*')
    hold on, plot(Zs{7}(1,:), Zs{7}(2,:),'y*')
    hold on, plot(Zs{8}(1,:), Zs{8}(2,:),'gd')
    hold on, plot(Zs{9}(1,:), Zs{9}(2,:),'rd')
    hold on, plot(Zs{10}(1,:), Zs{10}(2,:),'kd')
    title('block 1', 'FontSize',12,'FontWeight','Demi')

    subplot(1,2,2), plot(Zs{1}(3,:), Zs{1}(4,:),'*');
    hold on, plot(Zs{2}(3,:), Zs{2}(4,:),'r*')
    hold on, plot(Zs{3}(3,:), Zs{3}(4,:),'g*')
    hold on, plot(Zs{4}(3,:), Zs{4}(4,:),'k*')
    hold on, plot(Zs{5}(3,:), Zs{5}(4,:),'c*')
    hold on, plot(Zs{6}(3,:), Zs{6}(4,:),'m*')
    hold on, plot(Zs{7}(3,:), Zs{7}(4,:),'y*')
    hold on, plot(Zs{8}(3,:), Zs{8}(4,:),'gd')
    hold on, plot(Zs{9}(3,:), Zs{9}(4,:),'rd')
    hold on, plot(Zs{10}(3,:), Zs{10}(4,:),'kd')
    title('block 2', 'FontSize',12,'FontWeight','Demi')
end
keyboard
end