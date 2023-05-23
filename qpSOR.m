% Author : Rahul Choudhary

function [bestalpha] = qpSOR(Q, f, t, lb, ub, smallvalue)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Improved Pin TWSVM
%
%       bestalpha = qpSOR_new(Q, t, lb, ub, smallvalue)
%
%       Input:
%               Q     - Hessian matrix(Require positive definite).
%
%               f     - Vector for linear part of quadratic problem.
%
%               t     - (0,2) Paramter to control training.
%
%               lb    - Lower bound
%
%               ub    - Upper bound
%
%               zeroRange  - Number of alphas from start for which there are no constraints
%
%               smallvalue - Termination condition
%
%       Output:
%               bestalpha - Solutions of QPPs.
%
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initailization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[m,n]=size(Q);
alpha0=zeros(m,1);
L=tril(Q);
E=diag(Q);
twinalpha=alpha0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute alpha
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j=1:n
    twinalpha(j,1)=alpha0(j,1)-(t/E(j,1))*(Q(j,:)*twinalpha(:,1) + f(j) +L(j,:)*(twinalpha(:,1)-alpha0));
    if twinalpha(j,1)<lb(j)
        twinalpha(j,1)=lb(j);
    elseif twinalpha(j,1) > ub(j)
        twinalpha(j,1)=ub(j);
    else
        ;
    end
end

alpha=[alpha0, twinalpha];
iteration = 0;
while norm(alpha(:,2)-alpha(:,1))>smallvalue && iteration < 50
    for j=1:n
        twinalpha(j,1)=alpha(j,2)-(t/E(j,1))*(Q(j,:)*twinalpha(:,1) + f(j) +L(j,:)*(twinalpha(:,1)-alpha(:,2)));
        if twinalpha(j,1)<lb(j)
            twinalpha(j,1)=lb(j);
        elseif twinalpha(j,1)>ub(j)
            twinalpha(j,1)=ub(j);
        else
            ;
        end
    end
    iteration = iteration + 1;
    alpha(:,1)=[];
    alpha=[alpha,twinalpha];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bestalpha=alpha(:,2);
end
