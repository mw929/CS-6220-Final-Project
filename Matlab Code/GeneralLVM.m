function [G] = GeneralLVM(n)
%Author: Zachary

%Returns nxn rank 4 matrix generated by latent variable model, with
%non-orthognal functions

q2 = @(x) exp(-x.^2);
q3 = @(x) exp(-x);
q4 = @(x) (sin(x)).^2;
theta = unifrnd(0,1,n);
Lambda = diag(unifrnd(0,1,4,1));
Q = zeros(4,n);
for i = 1:1:n
    Q(1,i) = 1;
    Q(2,i) = q2(theta(i));
    Q(3,i) = q3(theta(i));
    Q(4,i) = q4(theta(i));
end
F = Q'*Lambda*Q;
G = F/max(max(abs(F)));