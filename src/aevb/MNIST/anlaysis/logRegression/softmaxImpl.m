function acc = softmaxImpl(train, test, nc, lambda)
% train, test data
%   each containing feature and label matrix
%   (N_example x F_dim) and(N_example x N_class)

    dim = size(train.X,2);
    w0 = rand(dim*(nc-1),1)-0.5;
    %lambda = 0.005;
    [ws, loss] = minimize(w0, @softmax, 50, train.X, train.Y, lambda);
    [labels, acc] = softmaxPredict(ws, test.X, test.Y);

end
