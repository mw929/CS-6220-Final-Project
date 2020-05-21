function T=BFS_tree(A,i_0,depth)
%A(input) : (N)x(N) weighted adjacency matrix of the data graph G = ([n],E')
%i_0(input): node to start with
%depth(input) : the depth of the layer required
%T(output): weighted directed adjacency matrix of the tree up to depth t

n=length(A);
L=cell(n,1);
for i=1:n
    L{i}=find(A(i,:)>0);%find adjacency nodes for node i
end
discovered=[i_0];%set the first discovered node to be the starting node
q=[i_0];%set the first number in the queue to be the starting node
T=sparse(n,n);%initialize the adjacency matrix for our tree
layer=0;%set the root layer to be 0
current_layer={[i_0]};
while not(isempty(q))
    
    temp={};
    for k=1:length(current_layer)
        j=q(1);
        q=q(2:length(q)); % pop the front
        nbh=L{j}; %let nbh denote the neighborhood of node q(1)
        
        for ii=1:length(nbh)
            if isempty(find(discovered==nbh(ii)))
                T(j,nbh(ii))=A(j,nbh(ii));
                %             T(nbh(ii),j)=1;
                temp{end+1}=[nbh(ii)];
                discovered=[discovered, nbh(ii)]; %append discovered node
                q=[q, nbh(ii)]; %append new discovered node to the queue
            end
        end
    end
    current_layer=temp;
    layer=layer+1;
    if layer==depth %stop at depth+1
        break
    end
    
end
end