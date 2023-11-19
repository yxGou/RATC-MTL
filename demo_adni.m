clc
clear
load('F:\A科研\CTC_ML\code\adni_pls\data\predict_data\X10_random_f.mat')
load('F:\A科研\CTC_ML\code\adni_pls\data\predict_data\Y10_random_f.mat')
MD=Y(:,:,1);
MA=Y(:,:,2);
MM=Y(:,:,3);
MRB=Y(:,:,5);
MR=Y(:,:,[4,6,7]);
MF=Y(:,:,8);

Y1(:,:,1)=MD;
Y1(:,:,2)=MA;
Y1(:,:,3)=MM;
Y1(:,:,4:6)=MR;
Y1(:,:,7)=MF;


% load('X1.mat')
% load('Y1.mat')

[n,T,V]=size(Y1);
Times=10;

RMSE=cell(1,Times);
CC=cell(1,Times);
NMSE=cell(1,Times);
WR=cell(1,Times);


RMSE_v=cell(1,Times);
CC_v=cell(1,Times);
NMSE_v=cell(1,Times);
WR_v=cell(1,Times);

P=cell(1,Times);

for k=1:Times
    inY=Y1;
    YC=Y1;
ratio_train=0.2;
for v=1:V
    Z=inY(:,:,v);
    indM=[];
    for t=1:T
         I=randperm(size(inY,1),ceil(size(inY,1)*ratio_train));
         indM=[indM;sort(I)];
    end
    for j=1:size(inY,2)
    Z(indM(j,:),j)=999;
    end
    inY(:,:,v)=Z;
end

indM=find(inY==999);% known data index
omega=find(inY~=999);% missing data index

YY=zeros(size(Y1))+999;
YY(indM)=Y1(indM);
Ytrain=cell(1,size(Y1,2)*size(Y1,3));
Ytest=cell(1,size(Y1,2)*size(Y1,3));
Xtrain=cell(1,size(Y1,2)*size(Y1,3));
Xtest=cell(1,size(Y1,2)*size(Y1,3));
Xo=cell(1,size(Y1,2)*size(Y1,3));
for v=1:size(Y1,3)
    for t=1:size(Y1,2)
    I1=find(YY(:,t,v)~=999);
    J1=find(YY(:,t,v)==999);
    Ytrain{(v-1)*size(Y1,2)+t}=Y1(I1,t,v);
    Xtrain{(v-1)*size(Y1,2)+t}=X(I1,:);
    Ytest{(v-1)*size(Y1,2)+t}=Y1(J1,t,v);
    Xtest{(v-1)*size(Y1,2)+t}=X(J1,:);
    Xo{(v-1)*size(Y1,2)+t}=J1;
    end
end

%% RATC-MTL
rho1=0.01;
rho2=0.1;
rho3=0.01;
alpha = [4, 1, 1e-3];
alpha = alpha / sum(alpha);
lambda=0.01;
rho=0.1;


[W_cFSGL, f] = Least_CFGLasso(Xtrain, Ytrain, rho1, rho2, rho3);
tasks=size(W_cFSGL,2);

YC(omega)=0;
[Yhat]=RATC(W_cFSGL,X,YC,indM,alpha,lambda,rho,200);


%% evaluate

A=zeros(T,V);
C=zeros(T,V);
B=zeros(1,V);
W=zeros(1,V);
p=zeros(T,V);
for v=1:V 
    s=0;N=0;b=0;
  for t=1:T
      A(t,v)=sqrt((norm((Y1(:,t,v)-Yhat(:,t,v)),2)^2)/n);
      s=s+(norm((Y1(:,t,v)-Yhat(:,t,v)),2)^2)/std(Y1(:,t,v));
      N=N+size(Y1,1);
      a=corrcoef(Y1(:,t,v),Yhat(:,t,v));
      C(t,v)=a(1,2);
      b=b+a(1,2)*size(Y1,1);
      [~,p(t,v)]=ttest(Y1(:,t,v),Yhat(:,t,v));
  end
  B(v)=s/N;
  W(v)=b/N;
end
RMSE{k}=A';
CC{k}=C';
NMSE{k}=B;
WR{k}=W;
P{k}=p;

Av=zeros(T,V);
Cv=zeros(T,V);
Bv=zeros(1,T);
Wv=zeros(1,T);
for t=1:T
    s=0;N=0;b=0;
  for v=1:V
      Av(t,v)=sqrt((norm((Y1(:,t,v)-Yhat(:,t,v)),2)^2)/n);
      s=s+(norm((Y1(:,t,v)-Yhat(:,t,v)),2)^2)/std(Y1(:,t,v));
      N=N+size(Y1,1);
      a=corrcoef(Y1(:,t,v),Yhat(:,t,v));
      Cv(t,v)=a(1,2);
      b=b+a(1,2)*size(Y1,1);
  end
  Bv(t)=s/N;
  Wv(t)=b/N;
end
RMSE_v{k}=Av';
CC_v{k}=Cv';
NMSE_v{k}=Bv;
WR_v{k}=Wv;

end





