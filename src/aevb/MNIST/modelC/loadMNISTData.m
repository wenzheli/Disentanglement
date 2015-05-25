function dataTr = loadMNISTdata()
% one extra operation is to renormalize the data

    load ../data/batchtraindata.mat;
    dataTr = batchdata';
    clear batchdata;

end
