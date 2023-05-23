%when cv and tuning are used
function [w1,b1,w2,b2,tot_time] = Truncated_pin_UTWSVM_func_(u01,u00,v01,v00,U,DataTrain,FunPara)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pin_UTWSVM: Universum Twin Support Vector Machine With truncated  Pinball Loss
%
% U=universum data
%u01=initial value regarding input data A.
%u00=initial value regrading kernel matrix A.
%v01=initial value regarding input data B.
%v00=initial value regrading kernel matrix B.
% 
%
%  Written by:Anuradha
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initailization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[~,no_col]=size(DataTrain);
obs = DataTrain(:,no_col);
A = DataTrain(obs==1,1:end-1);
B = DataTrain(obs~=1,1:end-1); 
c1=FunPara.c_1;
c2=c1;
cu=FunPara.c_u;
Uepsilon=FunPara.U_epsilon;
t1=FunPara.tau1;
t2=FunPara.tau2;
kerfPara = FunPara.kerfPara;
eps1 = 10^-4;
eps2 = 10^-4;
m1=size(A,1);
m2=size(B,1);
um=size(U,1);
e1=ones(m1,1);
e2=ones(m2,1);
eu=ones(um,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Kernel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(kerfPara.type,'lin')
    X1=[A e1];
    X2=[B e2];
    X3=[U eu];
else
   %%%%%%%%%%%using kernel function
    X=[A;B];
    X1=[(kernelfun(A,kerfPara,X)),e1];
    X2=[(kernelfun(B,kerfPara,X)),e2];
    X3=[(kernelfun(U,kerfPara,X)),eu];
end  
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute (w1,b1) and (w2,b2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tol=0.001;
som=1;
ite=0;

if strcmp(kerfPara.type,'lin')
    u0=u01;
else
    X=[A;B];
    AA=kernelfun(A,kerfPara,X);
    u0=u00;
end
w0=u0(1:end-1);
b0=u0(end);
s1=0.25;
delta10=zeros(1,size(B,1));
delta1=zeros(1,size(B,1));

%%%%CCCP procedure for the first hyperplane
if strcmp(kerfPara.type,'lin')
    F10=-(w0'*B'+b0);
 
else
    C=[A;B];
    P_10=kernelfun(B,kerfPara,C);
    F10=-(w0'*P_10'+b0);
end

for i=1:size(B,1)
        chk=1-F10(i)+s1;
        if chk > 0
            delta110(i)=c1*t1;
        end
    end
tic
while som>tol && ite<30
    ite=ite+1;    
    %%%%DtrunpinGTSVM1
    HH=X1'*X1;
    G=[X2;-X3];
    HH = HH + eps1*eye(size(HH));%regularization
    HHG = HH\G';
    kerH1=G*HHG;
    kerH1=(kerH1+kerH1')/2;
    f1=[e2;(-1+Uepsilon)*eu];
    lb=[-delta10';zeros(size(eu,1),1)];   
    ub=[c1*e2*(1+t2)-delta10';cu*eu];
    alpha1=qpSOR(kerH1,-f1,0.5,lb,ub,0.05);
    vpos=-HHG*alpha1;
    w1=vpos(1:end-1);
    b1=vpos(end);
    
    if strcmp(kerfPara.type,'lin')
         F1=-(w1'*B'+b1);
        
    else     
        C=[A;B];
        P_1=kernelfun(B,kerfPara,C);
        F1=-(w1'*P_1'+b1);
    end      
    for i=1:size(B,1)
        chk=1-F1(i)+s1;
        if chk > 0
            delta1(i)=c1*t1;
        end
    end

    som=norm(delta1-delta10);
    delta10=delta1;
end
tot_time=toc;

som2=1;
ite2=0;

if strcmp(kerfPara.type,'lin')
    v0=v01;
else
    X=[A;B];
    BB=kernelfun(B,kerfPara,X);
    v0=v00;
end
w20=v0(1:end-1);
b20=v0(end);
s2=0.25;
delta20=zeros(1,size(A,1));
delta2=zeros(1,size(A,1));


%%%%CCCP procedure for the second hyperplane

for i = 1:size(A,1)
   fp20=w20'*A(i,:)'+b20;
   F20=[F20 fp20];
end
F20=(w20'*A'+b20);
F20;

if strcmp(kerfPara.type,'lin')
    F20=(w20'*A'+b20); 
else
    C=[A;B];
    P_20=kernelfun(A,kerfPara,C);
    F20=(w20'*P_20'+b20);
end
for j=1:size(A,1)
    chk2=1-F20(j)+s2;
    if chk2 > 0
        delta20(j)=c2*t2;
    end
end
delta20;

tic
while som2>tol && ite2<30
    ite2=ite2+1;
    QQ=X2'*X2;
    QQ=QQ + eps2*eye(size(QQ));%regularization
    H=[X1;-X3];
    QQP=QQ\H';
    kerH2=H*QQP;
    kerH2=(kerH2+kerH2')/2;
    f2=[e1;(-1+Uepsilon)*eu];
    lb2=[-delta20';zeros(size(eu,1),1)]; 
    ub2=[c2*e1*(1+t1)-delta20';cu*eu];
    gamma1=qpSOR(kerH2,-f2,0.5,lb2,ub2,0.05);
    vneg=QQP*gamma1;
    
    w2=vneg(1:end-1);
    b2=vneg(end);
    
    
    if strcmp(kerfPara.type,'lin')
         for i = 1:size(A,1)
        fp2=w2'*A(i,:)'+b2;
        F2=[F2 fp2];
    end
        
    else     
        C=[A;B];
        P_2=kernelfun(A,kerfPara,C);
        F2=(w2'*P_2'+b2);
    end 
    for i=1:size(A,1)
        chk2=1-F2(i)+s2;
        if chk2 > 0
            delta2(i)=c2*t2;
        end
    end
    delta2;
    som2=norm(delta2-delta20);
    delta20=delta2;
end
tot_time=tot_time+toc;
end
