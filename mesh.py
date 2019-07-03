
%Input Parameters
ro=150; %outer radius
ri=30; %radius of inclusion
nelem=10; %number of elements
meshrefinementfactor=5; %ratio of element sizes at outer and inner radius

q=meshrefinementfactor^(1./(nelem-1));