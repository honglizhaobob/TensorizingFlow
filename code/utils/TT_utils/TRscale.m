function [A] = TRscale(A,scale)
%Scaling tensor train
A{1} = scale*A{1};
