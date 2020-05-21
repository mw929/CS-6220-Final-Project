%Created on Wed Apr 29 17:21:49 2020

%@author: Zachary

function [M_1,M_2,M_3,M1_ind,E1u,E1v,E2u,E2v,E3u,E3v]= SampleSplitting(M, p)
n = length(M);
p_prime = p / (4 - p);
M_1 = zeros(n,n);
M_2 = zeros(n,n);
M_3 = zeros(n,n);
M1_ind = zeros(n,n);
labels = [1; 2; 3];
p_labels = [1 / 4; 1 / 4; 1 / 2];
E1u=[];E1v=[];E2u=[];E2v=[];E3u=[];E3v=[];

for u =1:n
    for v =1:n
        if M(u,v) ~= 0
            z = randsample(labels,1,true,p_labels);
            if z == 1
                M_1(u,v) = M(u,v);
                E1u=[E1u, u];
                E1v=[E1v, v];
            elseif z == 2
                M_2(u,v) = M(u,v);
                E2u=[E2u, u];
                E2v=[E2v, v];
            else
                M_3(u,v) = M(u,v);
                E3u=[E3u, u];
                E3v=[E3v, v];
            end
        end
    end
end

for u =1:n
    for v =1:n
        if M_1(u,v) ~= 0
            r = binornd(1,p_prime);
            if (r == 1)
                M1_ind(u,v) = M_1(u,v);
            end
        end
    end
end
end