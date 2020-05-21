function [F_hat,d_hat,G]=Iterative_Collaborative_Filtering(M,kappa)
% This is the main function to recovery F via ICF
% M(input): nxn observation matrix
% kappa(input): density parameter
% F_hat(output): nxn matrix of estimation
% d_hat(ouput): nxn matrix of distance
% G(output): directed graph corresponding to M'.

%parameters
n=length(M);
p = n^(-1 + kappa);
p_prime = p / (4 - p);
t = floor(1/kappa-1);
rho = kappa/2;
eta = n^(-1/2 * (kappa - rho));

%sample splitting part
[M_1,M_2,M_3,M1_ind,E1u,E1v,E2u,E2v,E3u,E3v]= SampleSplitting(M, p);
G=digraph(E1u,E1v);

%initialization
d_hat=zeros(n,n);
N_tilde=cell(n,1);
%compute the BFS tree for each node
for u=1:n
    [N_utilde,N_utildeplus1]=NBD(M_1,u,t);
    A=[N_utilde,N_utildeplus1];
    N_tilde{u}=A;
end
%compute distance
for u = 1:n
    for v=u+1:n
        A=N_tilde{u};
        B=N_tilde{v};
        N_utilde=A(:,1);
        N_utildeplus1=A(:,2);
        N_vtilde=B(:,1);
        N_vtildeplus1=B(:,2);
        d_hat(u,v)=abs((1/p_prime)...
            *(N_utilde-N_vtilde)'*(M_2+M1_ind)*(N_utildeplus1-N_vtildeplus1));
        d_hat(v,u)=d_hat(u,v);
    end
end
%estimate F_hat
F_hat=zeros(n,n);
for u=1:n
    for v=1:n
        total = 0;
        count = 0;
        for idx =1:length(E3u)
            
            if d_hat(u, E3u(idx)) < eta && d_hat(v, E3v(idx)) < eta
                total = total + M(E3u(idx),E3v(idx));
                count = count+1;
            end
        end
        
        if count > 0
            F_hat(u, v) = total / count;
        end
    end
end
end





























