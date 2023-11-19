function [Y]=RATC(W,X,Y,omega,alpha,lambda,rho,maxIter)
epsilon=1e-5;

F=Y;
dim=size(Y);
N=length(dim);
M=cell(ndims(Y),1);
T=M;

normY=norm(Y(:));
for n=1:N
    M{n}=Y;
    T{n}=zeros(dim);
end

errList = zeros(maxIter, 1);
Msum = zeros(dim);
Tsum = zeros(dim);
for k = 1: maxIter
    Msum = 0*Msum;
    Tsum = 0*Tsum;
    for i = 1:N
        M{i} = Fold(Pro2TraceNorm(Unfold(Y-T{i}/rho,dim, i), alpha(i)/rho), dim, i);
        Msum = Msum + M{i};
        Tsum = Tsum + T{i};
    end
    
    lastY=Y;
    Y1=(lambda*X*W+unfold(rho*Msum+Tsum,1))/(lambda+N*rho);
%     Y1= (Msum + beta*Ysum) / (ndims(T)*beta);
    Y=Fold(Y1,dim,1);
    Y(omega)=F(omega);
    
    for i = 1:N
        T{i} = T{i} + rho*(M{i} - Y);
    end
    
    errList(k) = norm(Y(:)-lastY(:)) / normY;
    if errList(k) < epsilon
        break;
    end
end

    errList = errList(1:k);
fprintf('RATC ends: total iterations = %d   difference=%f\n\n', k, errList(k));

    