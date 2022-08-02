

function uz=smooth_ica_fun(x,Nco,p)

%%%%% Estimates Smooth Components from linear mixtures
%%%%% USAGE:   uz = smooth_ica_fun(x,Nco,p)
%%%%%  where  x   are the input data where observations are arranged in rows.
%%%%%         Nco is the number of desired components.
%%%%%         p   is the smoothness factor (0<p<1)
%%%%%         uz   are the extracted smooth components in the same
%%%%%         arrangement as x.
%%%%%
%%%%%  Implements the paper:
%%%%%  Mitianoudis N., Stathaki T., Constantinides A.G. " Smooth Signal
%%%%%  Extraction from instantaneous mixtures ", IEEE Signal Processing Letters, 
%%%%%  Vol. 14, No. 4, pp. 271-274, April 2007.
%%%%%
%%%%%  Code by Nikolaos Mitianoudis, Imperial College London, 2006


%%%%% PCA STEP %%%%%%%%%%%%
[N M]=size(x);
x=x-mean(x')'*ones(1,M);
C=x*x'/M;
[V,D]=eig(C);
V=V*sqrt(inv(D));
z=(V')*x;

%%%%%%%%%%% Find smooth component %%%%%%%%%%
dz=conv2(z,[1 -1],'same');

n2=0.9;  %%%%%%  Learning rate for Lagrange Multiplier optimisation
pp=[];
I=eye(N);
w_old=0;
uz=[];b=[];

%%%%%%%%% Correct in case Nco>N
if Nco>N
    Nco=N;
end
    
%%%%%%%%%% Correct Smoothness factor %%%%%
if p>1
    p=1;
end


for i=1:Nco


w=ones(N,1);w=w/norm(w);


l=50;   %%%%% Initial value of Lagrange Multiplier (you may change this)

iter=0;
dw=.5;
   
while dw<0.999999 & iter<300
    w_old=w;
    iter=iter+1
    u=w'*z;
    du=w'*dz;
    Cdz=dz*dz'/size(u,2);
    % Calculate Derivatives of J1 
    dJ1=z*tanh(u)'/size(x,2);
    d2J1=mean(1-tanh(u).^2)*I;
    % Calculate Derivatives of J2
    J2=mean(du.^2)-p*mean(u.^2);
    dJ2=dz*du'/size(dz,2)-p*z*u'/size(z,2);    
    d2J2=Cdz-p*I;         
    tmp=z*u'/size(u,2);
    
    %%%%% Update the weights %%%%
    %w= -dJ1-d2J1*w+l*(dJ2-d2J2*w);
    w=w-pinv(d2J1-l*d2J2,0.0001)*(dJ1-l*dJ2);
    %%%%% Update Lagrange Multiplier %%%
    l=l+n2*J2;
    %%%%% Orthogonalise to previous components 
    if i>1 
         w=w -b*(b')*w;
    end
    w=w/norm(w);
    dw=abs(w'*w_old);   
    pp=[pp [norm(dw);l]];
   
 end
 b= [b w];
 uz=[uz  ;(w')*z];  
end

%%%%%% Plot smooth components
for i=1:Nco
subplot(Nco,1,i);plot(uz(i,:));title(['Component' int2str(i)]);
end
% %%%% Plot Convergence Graphs for weights and Lagrange Multipliers for all
% %%%% components
% subplot(Nco+2,1,Nco+1);plot(pp(1,:));
% subplot(Nco+2,1,Nco+2);plot(pp(2,:));

