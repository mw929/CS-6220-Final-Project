%Created on Wed Apr 29 15:22:54 2020

%@author: Zachary

function [M]=SignalMatrix(F, kappa)  % Generates signal matrix M in Paper
    n = length(F);
    p = n^(-1 + kappa);
    M = zeros(n,n);
    for u=1:n
        for v=1:n
            r = binornd(1,p);
            if (r == 1)
                M(u,v) = F(u,v);  % Add Noise here somehow, seems tricky as we need M[u,v] in [0,1]
            else
                M(u,v)=0;

            end
        end
    end
end


