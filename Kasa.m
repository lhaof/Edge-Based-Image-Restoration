function Par = Kasa(XY)

%--------------------------------------------------------------------------
%  
%     Simple algebraic circle fit (Kasa method)
%      I. Kasa, "A curve fitting procedure and its error analysis",
%      IEEE Trans. Inst. Meas., Vol. 25, pages 8-14, (1976)
%
%     Input:  XY(n,2) is the array of coordinates of n points x(i)=XY(i,1), y(i)=XY(i,2)
%
%     Output: Par = [a b R] is the fitting circle:
%                           center (a,b) and radius R
%
%--------------------------------------------------------------------------

n = size(XY,1);      % number of data points
Z = XY(:,1).*XY(:,1) + XY(:,2).*XY(:,2);

XY1 = [XY ones(n,1)];
P = XY1 \ Z;

Par = [P(1)/2 , P(2)/2 , sqrt((P(1)*P(1)+P(2)*P(2))/4+P(3))];

end   %  Kasa
