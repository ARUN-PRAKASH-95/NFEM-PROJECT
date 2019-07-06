ro=150; %outer radius
ri=30; %radius of inclusion
nelem=10; %number of elements
meshrefinementfactor=5; %ratio of element sizes at outer and inner radius
%
%ratio between element sizes of subsequent elements for a geometric series
q=meshrefinementfactor^(1./(nelem-1))
%size of first element
lelem=(ro-ri)*(1-q)/(1-meshrefinementfactor*q);
rnode=ri;
rnodes=[ri];
%loop over all elements
for i=1:nelem
        fprintf('%d',i)
        rnode=rnode+lelem;
        rnodes=[rnodes;rnode]    %append
        lelem=lelem*q;
end
%visualize location of nodes
plot(rnodes,zeros(1,nelem+1),'x')
xlabel('r [\mum]')
legend('nodes')