%Created on Wed Apr 29 14:48:12 2020

%@author: Zachary
function [F] = PolyLatentVarMat(n) 
    theta = rand(n,1);
    spectrum = diag([8/11;1/11]);
    q = zeros(2,n);
    q(1, :) = 1;
    q(2, :) = 2.*sqrt(3).*(theta-1/2);
    F = q'*spectrum*q;
end
    
   
