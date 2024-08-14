%% ME524 - Computer exercise 3
clc, clear, close all

load Gnom
Gnom = G11;
[Gu74, info74] = ucover(Gmm, Gnom, 7);
W2 = info74.W1;
%% 3.1
[B, A] = tfdata(Gnom,'v');

% 3.
% P = poly([0.8 0.9 0.95]);
P = poly([0.99 0.95 0.95]);

% Fixed parts
Hs = [1 -1];
Hr = [1];

% 4.
[R, S] = poleplace(B,A,Hr,Hs,P);

% 5.
Pcheck = conv(A,S)+conv(B,R);

% 6.
T1 = sum(P)/sum(B);
T2 = sum(R);
% T = T1 = T2 = P(1)/B(1) = R(1) since we have an integrator in the
% controller

T = [T1]

% 7.
figure(1)
subplot(2,2,1)
U1 = tf(conv(A,R)',P,Ts,'variable','z^-1');
step(U1,tf(10),'--r',tf(-10),'--r')
title('Control signal')

subplot(2,2,2)
CL = tf(conv(T,B),P,Ts,'variable','z^-1');
step(CL)
title('Tracking step response')
stepinfo(CL)

subplot(2,2,3)
bodemag(U1)
title('Output sensitivity function')

subplot(2,2,4)
Ss = tf(conv(A,S)',P,Ts,'variable','z^-1');
bodemag(Ss,tf(1,0.5),'--r')
title('Input sensitivity function')

%% 9. 
Hr = [1 1];
[R, S] = poleplace(B,A,Hr,Hs,P);

Pcheck = conv(A,S)+conv(B,R);

% 6.
T = sum(R)

% 7.
figure(2)
subplot(2,2,1)
step(U1)
hold on
U = tf(conv(A,R)',P,Ts,'variable','z^-1');
step(U,tf(10),'--r',tf(-10),'--r')
legend('Hr = 1','Hr = 1 + q^{-1}')
title('Control signal')

subplot(2,2,2)
step(CL)
hold on
CL = tf(conv(T,B),P,Ts,'variable','z^-1');
step(CL)
legend('Hr = 1','Hr = 1 + q^{-1}')
title('Tracking step response')
stepinfo(CL)

subplot(2,2,3)
bodemag(U1)
hold on
bodemag(U)
legend('Hr = 1','Hr = 1 + q^{-1}')
title('Output sensitivity function')

subplot(2,2,4)
bodemag(Ss)
hold on
Ss = tf(conv(A,S)',P,Ts,'variable','z^-1');
bodemag(Ss,tf(1,0.5),'--r')
legend('Hr = 1','Hr = 1 + q^{-1}')
title('Input sensitivity function')

%% Q parametrization
nq = 16;
rng(3)
Q0 = randn(1,nq);
W1 = 0.5;

Rq = @(Q) sumpol(R',conv(conv(A,Hr),conv(Hs,Q))); % R = R0 + A*Hr*Hs*Q
Sq = @(Q) sumpol(S',-conv(conv(B,Hs),conv(Hr,Q))); % S = S0 - B*Hs*Hr*Q
Uq = @(Q) tf(conv(A,Rq(Q)),P,Ts,'variable','z^-1'); % U = A*R/P
Tq = @(Q) sum(Rq(Q));
fun = @(Q) norm(Uq(Q),inf);

sensq = @(Q) tf(conv(A,Sq(Q)),P,Ts,'variable','z^-1'); % Ss = A*S/P
tauq = @(Q) 1-sensq(Q);%tf(conv(T,B),P,Ts,'variable','z^-1'); % Ts = 1-Ss
ineq = @(Q) [norm(0.5*sensq(Q),inf)-1 ; norm(W2*tauq(Q),inf)-1];
ineq = @(Q) [norm(0.5*sensq(Q),inf)-1];

eq = [];

const = @(Q) deal(ineq(Q), eq);
opts = optimoptions('fmincon','Algorithm','interior-point','Display','iter','MaxFunctionEvaluations',5e+05);
[Qopt,~,exitflag,~,~,~,~] = fmincon(fun,Q0,[],[],[],[],[],[],const,opts)

Rnew = sumpol(R',conv(conv(A,Hr),conv(Hs,Qopt)));
Snew = sumpol(S',-conv(conv(B,Hs),conv(Hr,Qopt)));
Pnew = sumpol(conv(A,Snew),conv(B,Rnew));
Tnew = sum(Rnew);

% plot
figure(3)
subplot(2,2,1)
U = tf(conv(A,Rnew),Pnew,Ts,'variable','z^-1');
step(U,tf(10),'--r',tf(-10),'--r')
title('Control signal')

subplot(2,2,2)
CL = tf(conv(Tnew,B),Pnew,Ts,'variable','z^-1');
step(CL)
title('Tracking step response')
stepinfo(CL)

subplot(2,2,3)
bodemag(U)
title('Output sensitivity function')

subplot(2,2,4)
Ss = tf(conv(A,Snew),Pnew,Ts,'variable','z^-1');
bodemag(Ss,tf(1,0.5),'--r')
title('Input sensitivity function')

save('Rnew')
save('Snew')
save('Tnew')


STR = join(compose('%d',typecast(Tnew,'int64')),',') ; 
clipboard ('copy', STR{1})

function p = sumpol(p1,p2)
    n1 = length(p1);
    n2 = length(p2);

    if n1 ~= n2
        if n1 < n2
            p1 = [p1 zeros(1,n2-n1)];
        else
            p2 = [p2 zeros(1,n1-n2)];
        end
    end
    p = p1 + p2;
end


