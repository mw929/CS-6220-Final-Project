function [A,X] = MC_ALM(M, lambda, omega, tol )
%   This function solves the following problem
%   
%   min ( lambda*|| P.*X ||_1 + ||A||_* )
%   s.t. M = P.*(A + X) and P.*X=0.
%  
% 
%   The method is from
%   "Robust Principal Component Analysis with Missing Data"
%   by Fanhua Shang, Yuanyuan Liu, James Cheng, and Hong Cheng
%
%  We solve it via Augmented Langrange Multiplier Method(ALM) by alternating
%  directions.
%  M(input): mxn observation matrix
%  lambda(input): regularizer parameter
%  omega(input): indices of observation
%  tol(input): tolerence of convergence
%  A(output): mxn recovered low rank matrix
%  X(output): mxn matrix of noise




if ~exist('M', 'var')
    error('No observation data provided.');
end

if ~exist('omega', 'var')
    error('No observation set provided.');
end

if ~exist('tol', 'var')
    tol = 10^-6;
end

iterations=10000;
mu=lambda;


stop_vals = zeros(iterations, 2);

P = zeros(size(M));
P(omega) = 1;
R = ones(size(P)) - P;

Y = zeros(size(M));
X = zeros(size(M));

for k = 1 : iterations

    %% Step 1, solve for A
    
    temp = M - X + (1/mu)*Y;
    [U, S, V] = svd(temp, 'econ');
    s = diag(S);
    ind = find(s <= lambda/mu);
    s(ind) = 0;
    ind = find(s > lambda/mu);    
    s(ind) = s(ind) - lambda/mu;
    S = diag(s);
    A = U*S*(V');
    
    %% Step 2, update E
    
    old_X = X;
    X_c = R.*(M - A + (1/mu)*Y);
    temp = P.*(M - A + (1/mu)*Y);
    X_omega = sign(temp).*max(abs(temp)-lambda/mu,0);
    X=X_omega+X_c;
    
    %% Step 3, update Y
    
    Y = Y + mu*(M - A - X);
    
    %% Check stopping criteria
    k
    stop_vals(k, 1) = norm(M - A - X, 'fro') / norm(M, 'fro');
    stop_vals(k, 2) = min(mu, sqrt(mu)) * norm(X - old_X, 'fro') / norm(M, 'fro');
    
    if ( stop_vals(k, 1) <= tol && stop_vals(k, 2) <= tol)
        break;
    end
 
end
    
    
end

