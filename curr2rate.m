function f_rate = curr2rate(x,wgain,g,I,d,receptors)
% Computes transfer function from unit current to unit firing rate by a
% nonlinear function
%

y=g.*(x-I).*(1+receptors*wgain);

if y~=0
    f_rate = y./(1-exp(-d.*y));
else
    f_rate=0;
end
end