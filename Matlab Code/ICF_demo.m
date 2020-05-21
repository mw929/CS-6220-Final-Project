%% CS 6220 Final Project Iterative Collaborative Filtering Demo
clear all;clc; 
%%
%parameters
n=5;
ite=9;
kappa=0.4;
nn=[5,10,20,40,70,110,160,220,300,390,490,600,720,850,1000,1200];
%initialization
time=[];
MSE_list=[];
err1_relative_list=[];
errmax_list=[];
%% 
for k=nn
    
    F=PolyLatentVarMat(k);
    n=length(F);
    M=SignalMatrix(F,kappa);
    tic
    [F_hat,d_hat,G]=Iterative_Collaborative_Filtering(M,kappa);
    t=toc;
    time=[time,t];
    D=abs(F-F_hat).^2;
    MSE = sum(D(:))/n^2;
    MSE_list=[MSE_list,MSE];
    errmax=max(max(abs(F-F_hat)));
    errmax_list=[errmax_list, errmax];
    err1_relative=sum(sum(abs(F-F_hat)))/sum(sum(F));
    err1_relative_list=[err1_relative_list,err1_relative];
    
    
end

%% plots
figure
plot(nn(2:end),time(2:end),'x-');
title('Running Time for \kappa = 0.4');
xlabel('n');ylabel('time (s)');

figure
loglog(nn(2:end),time(2:end),'x-');
title('loglog plot of running time for \kappa = 0.4');
xlabel('n');ylabel('time (s)');

figure
plot(nn(2:end),MSE_list(2:end),'x-');
title('MSE for \kappa = 0.4');
xlabel('n');ylabel('MSE');

figure
plot(nn(2:end),err1_relative_list(2:end),'x-');
title('Relative Entrywise 1-norm Error for \kappa = 0.4');
xlabel('n');


figure
plot(nn(2:end),errmax_list(2:end),'x-');
title('Maximum Entrywise Error for \kappa = 0.4');
xlabel('n');

%% ICF vs. RMC
T=readtable('jester-data-1.xls');%load the data
raw_data=table2array(T);
idx=find(raw_data(:,1)==100);
A=raw_data(idx,2:end);

%%
n=100;
kappa=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45];
tol=10^-6;
lambda=1/sqrt(n);
A_100=A(1:n,1:n);
maxval=max(max(A_100));
minval=min(min(A_100));
A_N= (A_100-minval)./(maxval-minval);
F=[zeros(n,n),A_N;A_N',zeros(n,n)];
%%
MSE1=[];
MSE2=[];
MSE3=[];
MSE4=[];
for k=kappa
    M=SignalMatrix(F,k);
    [F_hat,d_hat,G]=Iterative_Collaborative_Filtering(M,k);
     F_hat_100=F_hat(1:n,n+1:2*n);
%      F_hat_100_back=F_hat_100.*(maxval-minval)+minval;
    MSE_N_ICF=1/n^2*sum(sum((A_N-F_hat_100).^2));
%     MSE_ICF=1/n^2*sum(sum((A_100-F_hat_100_back).^2))
    MSE1=[MSE1, MSE_N_ICF];
%     MSE2=[MSE2, MSE_ICF];

    M=SignalMatrix(A_N,k);
    [F_hat,d_hat,G]=Iterative_Collaborative_Filtering(M,k);
    MSE_N_ICF=1/n^2*sum(sum((A_N-F_hat).^2));
    MSE2=[MSE2, MSE_N_ICF];
%     M2=SignalMatrix(A_N,k);
%     omega=find(M2);
%     [M_hat,X_hat] = MC_ALM(M2, lambda, omega, tol );
%     for i=1:n
%         for j=1:n
%             if M_hat(i,j)<0
%                 M_hat(i,j)=mean(M_hat(:,j));
%             end
%         end
%     end
%     M_hat_100_back=M_hat.*(maxval-minval)+minval;
%     MSE_N_RMC=1/n^2*sum(sum((A_N-M_hat).^2))
%     MSE_RMC=1/n^2*sum(sum((A_100-M_hat_100_back).^2))
%     MSE3=[MSE3, MSE_N_RMC];
%     MSE4=[MSE4, MSE_RMC];
end






