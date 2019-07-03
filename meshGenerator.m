%% Generate list of position of nodes according to a geometric series
%    for assignement in "Nonlinear Finite Element Methods" 
%    in summer term 2019
%    lecturer in charge: Dr. Geralf Huetter
%
%Input Parameters
ro=150; %outer radius
ri=30; %radius of inclusion
nelem=10; %number of elements
meshrefinementfactor=5; %ratio of element sizes at outer and inner radius
%
%ratio between element sizes of subsequent elements for a geometric series
q=meshrefinementfactor^(1./(nelem-1));
%size of first element
lelem=(ro-ri)*(1-q)/(1-meshrefinementfactor*q);
rnode=ri;
rnodes=[ri];
%loop over all elements
for i=1:nelem
        rnode=rnode+lelem;
        rnodes=[rnodes;rnode];
        lelem=lelem*q;
end
%visualize location of nodes
plot(rnodes,zeros(1,nelem+1),'x')
xlabel('r [\mum]')
legend('nodes')
