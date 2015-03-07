function [W, objs, tmp] = admm(X, lam1, lam2, rho)

%  X :   N by D matrix, each row is the data point

[N,D] = size(X);  % N: # of training data  D: # of dimensions
K = 5;           % K: # of hidden units. 

X=X-mean(X(:));
X=X/std(X(:));

xhat = mean(X,1);
xstd = std(X,1)
X = X - ones(1, N)' * xhat;  
S = (1/N)*X'*X;   %  sample covariance matrix

L = pca(X);       %  initialize W using PCA
W = L(1:K,:);     
U1 = W*S*W';
U2 = W*S*W';
Z1 = U1;
Z2 = U2;

Max_Iter = 100;
objs = zeros(Max_Iter, 1);
rho_inv = 1/rho;

step_size = 0.00000005;
for i=1:Max_Iter
    
    % update objective function
    for j=1:80
       grad = (8*W*S*W'+ 2*(rho_inv*U1-Z1)' + 2*(rho_inv*U1-Z1) + 2*(rho_inv*U2-Z2)' + 2*(rho_inv*U2-Z2) + 4*W*W' - 4)*W*S + 6 *W;
       W = W - step_size * grad;
       %tmp = W*S*W';
       %(1/N)*norm(X-X*W'*W, 'fro')^2 + lam1*sum(svd(tmp)) + lam2*sum(abs(tmp(:))) + norm(W, 'fro')^2
    end
   
    % update Z1
    Z1 = W*S*W'+rho_inv*U1;
    [U, Q, V] = svd(Z1);
    Q(logical(eye(size(Q)))) = (diag(Q) > lam1/rho).*(diag(Q) - lam1/rho) ; 
    Zl = U*Q*V';
    
    % update Z2  
    tmp = W*S*W' + rho_inv * U2;
    Z2 = (tmp-lam2/rho).* (tmp > lam2/rho)
    Z2 = Z2+(tmp+lam2/rho).* (tmp <= -lam2/rho)
     
    % update U1,U2
    U2 = U2 + (W*S*W' - Z2);
    U1 = U1 + (W*S*W' - Zl);
    
    tmp = W*S*W';
    objs(i+1) = (1/N)*norm(X-X*W'*W, 'fro')^2 + lam1*sum(svd(tmp)) + lam2*sum(abs(tmp(:))) + norm(W, 'fro')^2;
    objs(i+1)
    
    lam1*sum(svd(tmp))
    lam2*sum(abs(tmp(:)))
end
    
