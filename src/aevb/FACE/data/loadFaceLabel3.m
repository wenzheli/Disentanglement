function labelBrightness = loadFaceLabel3()
% one extra operation is to renormalize the data
    
    load ../data/face.mat
    labelTr{1} = single(data(:,end-1));
    labelTr{2} = single(data(:,end));
    labelTr{3} = labelTr{1}*0;
    dataTr = single(data(:,1:end-2))';
    
    % for every combination of (id X view)
    % order the average pixel and divide all images into 3 groups
    [~, cat1] = hist(labelTr{1}, unique(labelTr{1}));
    [~, cat2] = hist(labelTr{2}, unique(labelTr{2}));
    
    for i=1:length(cat1)
        for j=1:length(cat2)
            idx = find(labelTr{1}==cat1(i) & labelTr{2}==cat2(j));
            if(sum(abs(labelTr{3}(idx)))>0)
                error('mistakes in previous operations');
            end

            if(~isempty(idx))
                d = dataTr(:,idx);
                v1 = mean(d); % average brightness
                v2 = sort(v1);
                n = length(v1);
                m = max(floor(n/3),1);
                border1 = v2(m);
                border2 = v2(n-m+1);
                subidx1 = find(v1<=border1);
                subidx2 = find(v1>=border2);
                subidx3 = find(v1>border1 & v1<border2);
                labelTr{3}(idx(v1<=border1)) = 1;
                labelTr{3}(idx(v1>=border2)) = 2;
                labelTr{3}(idx(v1>border1 & v1<border2)) = 3;
            end
        end
    end

    labelBrightness = labelTr{3};

end
