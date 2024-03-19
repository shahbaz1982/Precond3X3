clc
clear all
close all

global N
N=2^6 % size of the Matrix
K=9e-12;
[A,BB,D,CC]=CoMat(K,N);

[n nnA]=size(A);
[mC mmC]=size(CC);
[p ppD]=size(D);
B=BB';
C=CC';
[m mmB]=size(B);

Au = [A, B', C'; 
      B, zeros(m, m), zeros(m, p); 
      C, zeros(p, m), -D];

  % Construct the right-hand side vector b
f = rand(n, 1); % Example vector
g = rand(m, 1); % Example vector
h = rand(p, 1); % Example vector
b = [f; g; h];

 %----------------------0-GMRES Solver---------------------------------------
tol = 1e-8; % tolerance for FGMRES
restart = 10; % restart parameter for FGMRES

% Initialize FGMRES variables
[Ar Ac]=size(Au);
x0 = zeros(Ar, 1);
maxit = 5; % maximum number of iterations for FGMRES

% FGMRES solver
tic;

[x0, flag0, relres0, iter0, resvec0] = gmres(Au, b, restart, tol, maxit, [], [], x0);

semilogy(resvec0,'-*')
 fprintf('Preconditioner     Flag       RES              ITER:O(I)\n')
 fprintf('%d                   %d        %d       %d     %d\n',0,flag0,relres0,iter0)

%---------------------1-PR1 Preconditioner---------------------------------------
SA=B*inv(A)*B';
SB=B*inv(A)*C';
SC=D+C*inv(A)*C';

Q1  = [eye(n,n), zeros(n, m) zeros(n, p); 
      B*inv(A), eye(m, m), zeros(m, p); 
      C*inv(A), zeros(p, m), eye(p, p)];
  
Q2  = [A, zeros(n, m) zeros(n, p); 
      zeros(m, n), SA, zeros(m, p); 
      zeros(p, n), zeros(p, m), SC];

 Q3  = [eye(n,n), inv(A)*B', inv(A)*C'; 
      zeros(m, n), -eye(m, m), zeros(m, p); 
      zeros(p, n), zeros(p, m), -eye(p, p)];
  
  

PR1  = Q1*Q2*Q3;

[x1, flag1, relres1, iter1, resvec1] = gmres(Au, b, restart, tol, maxit, PR1, [], x0);

hold on
semilogy(resvec1,'-*')
 hold off
fprintf('%d                   %d        %d       %d     %d\n',1,flag1,relres1,iter1)

%----------------------2-PR2 Preconditioner---------------------------------------
Q4  = [A, zeros(n, m) zeros(n, p); 
      zeros(m, n), SA, zeros(m, p); 
      zeros(p, n), zeros(p, m), D];

PR2  = Q1*Q4*Q3;

[x2, flag2, relres2, iter2, resvec2] = gmres(Au, b, restart, tol, maxit, PR2, [], x0);

hold on
semilogy(resvec2,'-*')
 hold off
fprintf('%d                   %d        %d       %d     %d\n',2,flag2,relres2,iter2)

%----------------------3-PBT1 Preconditioner---------------------------------------

PBT1  = [A, B', C'; 
      zeros(m, n), -SA, zeros(m, p); 
      zeros(p, n), zeros(p, m), -SC];


[x3, flag3, relres3, iter3, resvec3] = gmres(Au, b, restart, tol, maxit, PBT1, [], x0);

hold on
semilogy(resvec3,'-o')
 hold off
fprintf('%d                   %d        %d       %d     %d\n',3,flag3,relres3,iter3)

 %---------------------4-PBT2 Preconditioner---------------------------------------

PBT2  = [A, B', C'; 
      zeros(m, n), -SA, -SB; 
      zeros(p, n), zeros(p, m), -SC];


[x4, flag4, relres4, iter4, resvec4] = gmres(Au, b, restart, tol, maxit, PBT2, [], x0);

hold on
semilogy(resvec4,'-o')
 hold off
fprintf('%d                   %d        %d       %d     %d\n',4,flag4,relres4,iter4)

 %----------------------5-PAPSS Preconditioner---------------------------------------
alpha1=0.001; 
PAPSS  = 0.5*[alpha1*eye(n,n)+A, B', C'; 
      -B, alpha1*eye(m,m), zeros(m,m); 
      -(1/alpha1)*C*(alpha1*eye(n,n)+A), (1/alpha1)*C*B', alpha1*eye(p,p)+D];


[x5, flag5, relres5, iter5, resvec5] = gmres(Au, b, restart, tol, maxit, PAPSS, [], x0);

hold on
semilogy(resvec5,'-o')
 hold off
fprintf('%d                   %d        %d       %d     %d\n',5,flag5,relres5,iter5)

 %-----------------------6-PRAPSS Preconditioner---------------------------------------
alpha2=0.01;
PRAPSS  = [A, B', C'; 
      -B, alpha2*eye(m,m), zeros(m,m); 
      -(1/alpha2)*C*A, (1/alpha2)*C*B', D];


[x6, flag6, relres6, iter6, resvec6] = gmres(Au, b, restart, tol, maxit, PRAPSS, [], x0);

hold on
semilogy(resvec6,'-o')
 hold off
fprintf('%d                   %d        %d       %d     %d\n',6,flag6,relres6,iter6)
%------------------------7-PTPBT Preconditioner---------------------------------------
alpha=1.3333e-5;
beta=7.1429e-2;
PTPBT  = [A, B', C'; 
       B     -alpha*eye(m,m),  zeros(m, p); 
      zeros(p, n), zeros(p, m), -(D+beta*C*C')];


[x7, flag7, relres7, iter7, resvec7] = gmres(Au, b, restart, tol, maxit, PTPBT, [], x0);

hold on
semilogy(resvec7,'-+')
 hold off
fprintf('%d                   %d        %d       %d     %d\n',7,flag7,relres7,iter7)

%--------------------------8-PCR1 Preconditioner---------------------------------------

PCR1  = [A, B', zeros(n, p); 
      -B, zeros(m, m), zeros(m, p); 
      -C, zeros(p, m), -D];


[x8, flag8, relres8, iter8, resvec8] = gmres(Au, b, restart, tol, maxit, PCR1, [], x0);

hold on
semilogy(resvec8,'-+')
 hold off
fprintf('%d                   %d        %d       %d     %d\n',8,flag8,relres8,iter8)

 %---------------------------9-PCR2 Preconditioner------------------------------------------
DG=diag(D);
DGG=zeros(p,p);
for i=1:p
    DGG(i,i)=DG(i,1);
end

PCR2  = [A+C'*inv(DGG)*C, B', zeros(n, p); 
      -B, zeros(m, m), zeros(m, p); 
      -C, zeros(p, m), D];

[x9, flag9, relres9, iter9, resvec9] = gmres(Au, b, restart, tol, maxit, PCR2, [], x0);
hold on
semilogy(resvec9,'-+')
hold off
fprintf('%d                   %d        %d       %d     %d\n',9,flag9,relres9,iter9)
%-----------------------10-PMR1 Preconditioner---------------------------------------
Q5  = [eye(n,n), zeros(n, m) zeros(n, p); 
      B*inv(A), eye(m, m), zeros(m, p); 
      C*inv(A), SB'*inv(SA), eye(p, p)];
  
Q6  = [eye(n,n), inv(A)*B', inv(A)*C'; 
      zeros(m, n), -eye(m, m), -inv(SA)*SB; 
      zeros(p, n), zeros(p, m), -eye(p, p)];
  
PMR1  = Q5*Q2*Q6;

[x10, flag10, relres10, iter10, resvec10] = gmres(Au, b, restart, tol, maxit, PMR1, [], x0);

hold on
semilogy(resvec10,'-*')
 hold off
fprintf('%d                  %d        %d       %d     %d\n',10,flag10,relres10,iter10)

%--------------------------11-PMR2 Preconditioner---------------------------------------

PMR2  = Q5*Q4*Q6;

[x11, flag11, relres11, iter11, resvec11] = gmres(Au, b, restart, tol, maxit, PMR2, [], x0);

hold on
semilogy(resvec11,'-*')
 hold off
fprintf('%d                  %d        %d       %d     %d\n',11,flag11,relres11,iter11)

 %---------------------------12-P Preconditioner------------------------------------------

P  = [A-C'*inv(D)*C, B', C'; 
      B, zeros(m, m), zeros(m, p); 
      2*C, zeros(p, m), -D];


[x12, flag12, relres12, iter12, resvec12] = gmres(Au, b, restart, tol, maxit, P, [], x0);
hold on
semilogy(resvec12,'-+')
 
legend('GMRES','PR1','PR2','PBT1','PBT2','PAPSS','PRAPSS','PTPBT','PCR1','PCR2','PMR1','PMR2','P')
%title('Relative Residual Norms')
hold off
fprintf('%d                  %d        %d       %d     %d\n',12,flag12,relres12,iter12)

%--------------------------------Eigenvalues Display------------------------------------------
figure;%A
eigenvalues_A = eig(full(Au));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix A ');
xlabel('Real Part');
ylabel('Imaginary Part');

figure;%PR1^{-1}A
eigenvalues_A = eig(full(inv(PR1)*Au));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix P_{R1}^{-1} A ');
xlabel('Real Part');
ylabel('Imaginary Part');

figure;%PR2^{-1}A
eigenvalues_A = eig(full(inv(PR2)*Au));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix P_{R2}^{-1} A ');
xlabel('Real Part');
ylabel('Imaginary Part');

figure;%PBT1^{-1}A
eigenvalues_A = eig(full(inv(PBT1)*Au));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix P_{BT1}^{-1} A ');
xlabel('Real Part');
ylabel('Imaginary Part');

figure;%PBT2^{-1}A
eigenvalues_A = eig(full(inv(PBT2)*Au));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix P_{BT2}^{-1} A ');
xlabel('Real Part');
ylabel('Imaginary Part');

figure;%PAPSS^{-1}A
eigenvalues_A = eig(full(inv(PAPSS)*Au));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix P_{APSS}^{-1} A ');
xlabel('Real Part');
ylabel('Imaginary Part');

figure;%PRAPSS^{-1}A
eigenvalues_A = eig(full(inv(PRAPSS)*Au));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix P_{RAPSS}^{-1} A ');
xlabel('Real Part');
ylabel('Imaginary Part');

figure;%A
eigenvalues_A = eig(full(inv(PTPBT)*Au));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix P_{TPBT}^{-1} A ');
xlabel('Real Part');
ylabel('Imaginary Part');

figure;%PCR1^{-1}A
eigenvalues_A = eig(full(inv(PCR1)*Au));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix P_{CR1}^{-1} A ');
xlabel('Real Part');
ylabel('Imaginary Part');

figure;%PCR2^{-1}A
eigenvalues_A = eig(full(inv(PCR2)*Au));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix P_{CR2}^{-1} A ');
xlabel('Real Part');
ylabel('Imaginary Part');

figure;%PMR1^{-1}A
eigenvalues_A = eig(full(inv(PMR1)*Au));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix P_{MR1}^{-1} A ');
xlabel('Real Part');
ylabel('Imaginary Part');

figure;%PMR2^{-1}A
eigenvalues_A = eig(full(inv(PMR2)*Au));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix P_{MR2}^{-1} A ');
xlabel('Real Part');
ylabel('Imaginary Part');

figure;%P^{-1}A
eigenvalues_A = eig(full(inv(P)*Au));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix P^{-1} A ');
xlabel('Real Part');
ylabel('Imaginary Part');

%--------------------------Function----------------------------
function [A,B,C,D]=CoMat(K,N)
%clear all
%format short e

global n Np1 dz dzi alsq beta 
global u v w phi lambda 
global precon solver msolve

% LC parameters
%K=9e-12;
eps_0=8.8542e-12; 
eps_par=15; eps_perp=5; eps_a=eps_par-eps_perp;
alphac=sqrt(3)*pi/2; % 2.721
beta=eps_perp/eps_a;

% set desired alpha
%alpha=0.5*alphac; % no electric field for Off-state
alpha=1.5*alphac; % On-state
alsq=alpha^2;

% find resulting voltage
V=sqrt((K*alsq)/(eps_0*eps_a));

fprintf('alpha: %8.4e, beta: %8.4e, voltage: %8.4e\n', ...
         alpha,beta,V)
 
% set grid size N
%N=2^2;
fprintf('\n N=%3i\n', N);

% construct coefficient matrix
solver=0; % 0 for full matrix, 1 for reduced matrix
[H,A,B,C,D]= mgen(V);
end

function [H,A,B,C,D]=mgen(V)
% constructs initial matrix for 1D TN test problem

global n N Np1 dz dzi alsq beta 
global u v w phi lambda
global precon solver msolve

% number of unknowns
n=N-1; Np1=N+1;

% discretisation parameter
dz=1/N; dzi=1/dz;

% coordinates 
x=ones(Np1,1); y=ones(Np1,1); z=linspace(0,1,Np1)';

% choose initial guess 
init=2;
if init==1 
   % pure twist
   angle=0; theta=angle*ones(Np1,1); psi=linspace(0,pi/2,Np1)';
   u=cos(theta).*cos(psi); v=cos(theta).*sin(psi); w=sin(theta);
   phi=linspace(0,1,Np1)'; 
elseif init==2
   % twist and tilt
   theta=sin(pi*z);  psi=linspace(0,pi/2,Np1)';  % top curve 
%  theta=sin(-pi*z); psi=linspace(0,-pi/2,Np1)'; % bottom curve 
   u=cos(theta).*cos(psi); v=cos(theta).*sin(psi); w=sin(theta);
   phi=linspace(0,1,Np1)'; 
end

% inital guess for lambda
initl=2;
if initl==1 
   % pure twist
   lambda=-pi^2/4*dz*ones(Np1,1); lambda(1)=0; lambda(end)=0;
elseif initl==2
   % twist and tilt
   lambda=dz*alsq*ones(Np1,1); lambda(1)=0; lambda(end)=0;
end 

% apply boundary conditions
u(1)=1;v(1)=0;w(1)=0;u(end)=0;v(end)=1;w(end)=0;  % top curve
%u(1)=1;v(1)=0;w(1)=0;u(end)=0;v(end)=-1;w(end)=0; % bottom curve
phi(1)=0; phi(end)=1; 

% construct full solution vector
nvec=[u; v; w];
sol=[nvec; lambda; phi];

% construct Hessian
[H,A,B,C,D]=set_hessian;

if solver==1
   % construct nullspace matrix
   val=sqrt(v(2:N).^2+w(2:N).^2);
   Nmat(1:n,n+1:2*n)=spdiags(val,0,n,n);
   Nmat(n+1:2*n,1:n)=spdiags(-w(2:N)./val,0,n,n);
   Nmat(n+1:2*n,n+1:2*n)=spdiags(-(v(2:N).*u(2:N))./val,0,n,n);
   Nmat(2*n+1:3*n,1:n)=spdiags(v(2:N)./val,0,n,n);
   Nmat(2*n+1:3*n,n+1:2*n)=spdiags(-(w(2:N).*u(2:N))./val,0,n,n);
   % construct rhs
   [grad,grad_n,grad_l,grad_p]=setup_gradient; rhs=-grad;
   % set up reduced system without B
   xhat=-B*((B'*B)\grad_l);
   temp1=Nmat'*D;
   temp2=Nmat'*A*Nmat;
   H=[temp2 temp1; temp1' -C];
end


end

function [Hhat,Ahat,Bhat,Chat,Dhat]=set_hessian
% sets up Hessian with boundary equations ignored

global n N Np1 dz dzi alsq beta 
global u v w phi lambda

os=ones(n,1); 

% set up midpoint values
wsqhalf=(w(2:Np1).^2+w(1:N).^2)/2;
ahalf=alsq*(beta+wsqhalf);
phidiff=phi(2:Np1)-phi(1:N);

% set up Ahat
vc=2+dz*lambda(2:N); 
vr=-os; vl=-os;
A11=dzi*spdiags([vl vc vr],-1:1,n,n);
A22=A11;
vp=phidiff(1:n).^2+phidiff(2:N).^2;
A33=A11-dzi*alsq/2*spdiags(vp,0,n,n);
%Ahat=blkdiag(A11,A22,A33);
Ahat(1:n,1:n)=A11;
Ahat(n+1:2*n,n+1:2*n)=A22;
Ahat(2*n+1:3*n,2*n+1:3*n)=A33;

% set up Bhat
BU=spdiags(u(2:N),0,n,n); 
BV=spdiags(v(2:N),0,n,n); 
BW=spdiags(w(2:N),0,n,n); 
%Bhat=[BU; BV; BW];
Bhat(1:n,1:n)=BU;
Bhat(n+1:2*n,1:n)=BV;
Bhat(2*n+1:3*n,1:n)=BW;

% set up Chat
vc=ahalf(1:n)+ahalf(2:N);
vl=[-ahalf(2:n); 0]; vr=[0; -ahalf(2:n)]; 
Chat=dzi*spdiags([vl vc vr],-1:1,n,n);

%keyboard
% set up Dhat
mul=alsq*dzi*spdiags(w(2:N),0,n,n);
vc=phi(1:n)-2*phi(2:N)+phi(3:Np1);
vl=[phidiff(2:n);0]; vr=[0;-phidiff(2:n)];
DW=mul*spdiags([vl vc vr],-1:1,n,n);
Dhat=spalloc(3*n,n,3*n);
Dhat(2*n+1:3*n,:)=DW;

%e=ones(n,1); z=zeros(n,1); mat=alsq*spdiags([e z -e],-1:1,n,n);

%ev=eig(full(inv(Chat)));
%nC=norm(full(inv(Chat)));
%disp([nC nC*dz nC/dz])

% set up Hhat
temp=[Bhat Dhat];
tC=spalloc(2*n,2*n,3*n);
tC(n+1:2*n,n+1:2*n)=-Chat;
%Hhat=[Ahat temp; temp' tC];
Hhat(1:3*n,1:3*n)=Ahat;
Hhat(1:3*n,3*n+1:5*n)=temp;
Hhat(3*n+1:5*n,1:3*n)=temp';
Hhat(3*n+1:5*n,3*n+1:5*n)=tC;


end
