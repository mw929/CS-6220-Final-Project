function [N_tilde,N_tildeplus1]=NBD(A,i_0,depth)
% A(input) : (N)x(N) weighted adjacency matrix of the data graph G = ([n],E')
% i_0(input): node to start with
% depth(input) : the depth of the layer required
% N_tilde(output): Nx1 sparse vector of neighbor boundary at depth t
% N_tildeplus1(output): Nx1 sparse vector of neighbor boundary at depth t+1
n=length(A);
L=cell(n,1);
for i=1:n
    L{i}=find(A(i,:)>0);%find adjacency nodes for node i
end
discovered=[i_0];%set the first discovered node to be the starting node
q=[i_0];%set the first number in the queue to be the starting node
T=sparse(n,n);%initialize the adjacency matrix for our tree
N_tilde=sparse(n,1);
N_tildeplus1=sparse(n,1);
layer=0;%set the root layer to be 0
current_layer={[i_0,1]};
while not(isempty(q))
    
    temp={};
    for k=1:length(current_layer)
        j=q(1);
        q=q(2:length(q)); % pop the front
        nbh=L{j}; %let nbh denote the neighborhood of node q(1)
        for ii=1:length(nbh)
            if isempty(find(discovered==nbh(ii)))
                temp{end+1}=[nbh(ii),current_layer{k}(2)*A(j,nbh(ii))];
                discovered=[discovered, nbh(ii)]; %append discovered node
                q=[q, nbh(ii)]; %append new discovered node to the queue
            end
        end
    end
    current_layer=temp;
    layer=layer+1;
    if layer==depth
        if length(current_layer)~=0
            % compute normalized N_tilde at depth t
            nn=length(current_layer);
            for kk=1:nn
                indx=current_layer{kk}(1);
                val=current_layer{kk}(2);
                N_tilde(indx)=val/nn;
            end
            
        end
    elseif layer==depth+1
        if length(current_layer)~=0
            % compute normalized N_tilde at depth t+1
            nn=length(current_layer);
            for kk=1:nn
                indx=current_layer{kk}(1);
                val=current_layer{kk}(2);
                N_tildeplus1(indx)=val/nn;
            end
            
        end
        break
    end
end
end