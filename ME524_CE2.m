%% Computer exercise 2
clc, clear, close all

%% 2.1
% logs_1
load logs_1.mat
G1 = oe(data, [8 8 1]); % Parametric
Ts = G1.Ts; % Same for all data
freqs = (pi/4096:pi/4096:pi) / Ts; % Same for all data
Gf1 = spa(data,8191,freqs); % Non-parametric

opts = nyquistoptions;
opts.ConfidenceRegionDisplaySpacing = 3;
opts.ShowFullContour = 'off';

figure(1)
nyquistplot(Gf1,G1,freqs,opts,'sd',2.45);
legend('Non-parametric','Parametric')
% % add a zoomed zone
% box on
% zp = BaseZoom();
% zp.run;

% logs_3
load logs_3.mat
G3 = oe(data, [8 8 1]); % Parametric
Gf3 = spa(data,8191,freqs); % Non-parametric

opts = nyquistoptions;
opts.ConfidenceRegionDisplaySpacing = 3;
opts.ShowFullContour = 'off';

figure(2)
nyquistplot(Gf3,G3,freqs,opts,'sd',2.45);
legend('Non-parametric','Parametric')

% logs_5
load logs_5.mat
G5 = oe(data, [8 8 1]); % Parametric
Gf5 = spa(data,8191,freqs); % Non-parametric

opts = nyquistoptions;
opts.ConfidenceRegionDisplaySpacing = 3;
opts.ShowFullContour = 'off';

figure(3)
nyquistplot(Gf5,G5,freqs,opts,'sd',2.45);
legend('Non-parametric','Parametric')

% logs_7
load logs_7.mat
G7 = oe(data, [8 8 1]); % Parametric
Gf7 = spa(data,8191,freqs); % Non-parametric

opts = nyquistoptions;
opts.ConfidenceRegionDisplaySpacing = 3;
opts.ShowFullContour = 'off';

figure(4)
nyquistplot(Gf7,G7,freqs,opts,'sd',2.45);
legend('Non-parametric','Parametric')

% logs_9
load logs_9.mat
G9 = oe(data, [8 8 1]); % Parametric
Gf9 = spa(data,8191,freqs); % Non-parametric

opts = nyquistoptions;
opts.ConfidenceRegionDisplaySpacing = 3;
opts.ShowFullContour = 'off';

figure(5)
nyquistplot(Gf9,G9,freqs,opts,'sd',2.45);
legend('Non-parametric','Parametric')

% logs_11
load logs_11.mat
G11 = oe(data, [8 8 1]); % Parametric
Gf11 = spa(data,8191,freqs); % Non-parametric

opts = nyquistoptions;
opts.ConfidenceRegionDisplaySpacing = 3;
opts.ShowFullContour = 'off';

figure(6)
nyquistplot(Gf11,G11,freqs,opts,'sd',2.45);
legend('Non-parametric','Parametric')

% 2.1.5 To choose the nominal model we gonna plot the bode magnitude of the
% multimodel

% Bode diagram of all models
figure(7)
bodemag(G1,G3,G5,G7,G9,G11)
legend('G1','G3','G5','G7','G9','G11')

Gmm = stack(1,G1,G3,G5,G7,G9,G11);

% test all the nominal (by plotting the filter)
[Gu1, info1] = ucover(Gmm, G1, 2);
[Gu3, info3] = ucover(Gmm, G3, 2);
[Gu5, info5] = ucover(Gmm, G5, 2);
[Gu7, info7] = ucover(Gmm, G7, 2);
[Gu9, info9] = ucover(Gmm, G9, 2);
[Gu11,info11] = ucover(Gmm, G11, 2);

W2_1 = info1.W1opt;
W2_3 = info3.W1opt;
W2_5 = info5.W1opt;
W2_7 = info7.W1opt;
W2_9 = info9.W1opt;
W2_11 = info11.W1opt;

figure(8) % Gnom = G1
hold on
bodemag((G3/G1)-1, 'b')
bodemag((G5/G1)-1,'b')
bodemag((G7/G1)-1,'b')
bodemag((G9/G1)-1,'b')
bodemag((G11/G1)-1,'b')
bodemag(W2_1,'r')
title('Nominal model : G1')
legend('G3/G1)-1','(G5/G1)-1','G7/G1)-1','(G9/G1)-1','(G11/G1)-1','W2_1' )

figure(9) % Gnom = G3
hold on
bodemag((G1/G3)-1, 'b')
bodemag((G5/G3)-1,'b')
bodemag((G7/G3)-1,'b')
bodemag((G9/G3)-1,'b')
bodemag((G11/G3)-1,'b')
bodemag(W2_3,'r')
title('Nominal model : G3')

figure(10) % Gnom = G5
hold on
bodemag((G1/G5)-1, 'b')
bodemag((G3/G5)-1,'b')
bodemag((G7/G5)-1,'b')
bodemag((G9/G5)-1,'b')
bodemag((G11/G5)-1,'b')
bodemag(W2_5,'r')
title('Nominal model : G5')

figure(11) % Gnom = G7
hold on
bodemag((G1/G7)-1, 'b')
bodemag((G3/G7)-1,'b')
bodemag((G5/G7)-1,'b')
bodemag((G9/G7)-1,'b')
bodemag((G11/G7)-1,'b')
bodemag(W2_7,'r')
title('Nominal model : G7')

figure(12) % Gnom = G9
hold on
bodemag((G1/G9)-1, 'b')
bodemag((G3/G9)-1,'b')
bodemag((G5/G9)-1,'b')
bodemag((G7/G9)-1,'b')
bodemag((G11/G9)-1,'b')
bodemag(W2_9,'r')
title('Nominal model : G9')

figure(13) % Gnom = G11
hold on
bodemag((G1/G11)-1, 'b')
bodemag((G3/G11)-1,'b')
bodemag((G5/G11)-1,'b')
bodemag((G7/G11)-1,'b')
bodemag((G9/G11)-1,'b')
bodemag(W2_11,'r')
title('Nominal model : G11')

% All W2 in one plot
figure(14)
bodemag(W2_1,W2_3,W2_5,W2_7,W2_9,W2_11)
title('Comparison between different weighting filter')
legend('W2 (Gnom = G1)','W2 (Gnom = G3)','W2 (Gnom = G5)','W2 (Gnom = G7)','W2 (Gnom = G9)','W2 (Gnom = G11)','Location','northwest')

[Gu74, info74] = ucover(Gmm, G7, 4);
W2 = info74.W1;
Gnom = G7;
save('W2')
save('Gnom')
save('Gmm')
%% 2.2 
% Load data
load('Gnom')
load('Gmm')
load('W2')

% W1S < 1 for Gmm
% [W1s W2T] < sqrt(2)/2 for Gnom

% Design of W1 
num = [1 7];
den = [1 0.0001];
W1 = tf(num,den) * 1/2;
W1 = c2d(W1,Ts,'zoh');
figure(15)
bodemag(W1^-1,tf(2))

% Design of K (Hinf controller)
Kinf = mixsyn(Gnom,W1,[],W2);

T = feedback(Gmm*Kinf,1);
Tnom = feedback(Gnom*Kinf,1);
stepinfo(Tnom) % Check settling time
U =feedback(Kinf,Gmm);
Unom =feedback(Kinf,Gnom);
S = feedback(1,Gmm*Kinf);
Snom = feedback(1,Gnom*Kinf);   

% Plot
figure(16)
subplot(2,2,1)
step(T,Tnom)
title('Step response')
legend('Multimodel','Nominal')
subplot(2,2,2)
step(U,Unom)
title('Control signal')
legend('Multimodel','Nominal')
ylim([-1.5,1.5])
subplot(2,2,3)
bodemag(U,Unom)
xlim([10^-2 1560])
title('Sensitvity function U')
legend('Multimodel','Nominal')
subplot(2,2,4)
bodemag(S,Snom, W1^-1)
xlim([10^-2 1560])
title('Sensitivity function S')
legend('Multimodel','Nominal', 'W1 inv')

condition = norm([W1*Snom W2*Tnom],inf)
condition2 = norm(W1*S,inf)
if (condition <= 1/sqrt(2)) && (all(condition2 <= 1))
    fprintf('Robust performance conditions are met\n')
end
%%
% W3 to imit the magnitude of U
W3 = 0.1;
Kinf_range = mixsyn(Gnom, W1, W3, W2);

T = feedback(Gmm*Kinf_range,1);
Tnom = feedback(Gnom*Kinf_range,1);
U =feedback(Kinf_range,Gmm);
Unom =feedback(Kinf_range,Gnom);
S = feedback(1,Gmm*Kinf_range);
Snom = feedback(1,Gnom*Kinf_range);

% Plot
figure(17)
subplot(2,2,1)
step(T,Tnom)
title('Step response')
legend('Multimodel','Nominal')
subplot(2,2,2)
step(U,Unom)
title('Control signal')
legend('Multimodel','Nominal')
ylim([-1.5,1.5])
subplot(2,2,3)
bodemag(U,Unom,tf(1/W3))
title('Sensitivity function U')
legend('Multimodel','Nominal','W3^-1')
xlim([10^-2 1560])
subplot(2,2,4)
bodemag(S,Snom, W1^-1)
title('Sensitivity function S')
xlim([10^-2 1560])
legend('Multimodel','Nominal', 'W1 inv')

condition = norm([W1*Snom W2*Tnom],inf)
condition2 = norm(W1*S,inf)
if (condition <= 1/sqrt(2)) && (all(condition2 <= 1))
    fprintf('Robust performance conditions are met\n')
end
%%
% Reduce order of K
orderK = size(Kinf_range.A,1); 

K_ft = ss2tf(Kinf_range.A,Kinf_range.B,Kinf_range.C,Kinf_range.D);
Kreduced = reduce(Kinf_range, 11); % 11th-order
orderKreduced = size(Kreduced.A,1);

% Comparison pole-zero map
figure(18)
subplot(1,2,2)
pzmap(Kreduced)
title('Controller after reduction')
subplot(1,2,1)
pzmap(Kinf_range)
title('Controller before reduction')

% Comparison controllers
figure(19)
bodemag(Kinf_range,Kreduced)
legend('Controller before reduction', 'Controller after reduction')
title('Reduction of the order of the controller')

% Test performance
T = feedback(Gmm*Kreduced,1);
Tnom = feedback(Gnom*Kreduced,1);
U =feedback(Kreduced,Gmm);
Unom =feedback(Kreduced,Gnom);
S = feedback(1,Gmm*Kreduced);
Snom = feedback(1,Gnom*Kreduced);

figure(20)
subplot(2,2,1)
step(T,Tnom)
title('Step response')
legend('Multimodel','Nominal')
subplot(2,2,2)
step(U,Unom)
title('Control signal')
legend('Multimodel','Nominal')
ylim([-1.5,1.5])
subplot(2,2,3)
bodemag(U,Unom, 1/tf(W3))
xlim([10^-2 1560])
title('Sensitivity function U')
legend('Multimodel','Nominal','W3^-1')
subplot(2,2,4)
bodemag(S,Snom, W1^-1)
xlim([10^-2 1560])
title('Sensitivity function S')
sgtitle('Model-based H_\infty control design')
legend('Multimodel','Nominal', 'W1 inv')

condition1 = norm([W1*Snom W2*Tnom],inf)
condition2 = norm(W1*S,inf)
if (condition1 <= 1/sqrt(2)) && (all(condition2 <= 1))
    fprintf('Robust performance conditions is met\n')
end

% Infinity norm
norm_nominal = norm([W1*Snom W2*Tnom],inf)
norm_multimodel = norm([W1*S W2*T],inf)

% Modify report + report Datadriven
%% H2 controller
% Conversion from discrete time to continuous time
Gct = d2c(Gnom);
[A,B,C,D] = ssdata(Gct);

% Optimization problem definition
n = size(A,1);
m = size(B,2);

% Decision variables
L = sdpvar(n,n,'symmetric');
X = sdpvar(m,n);
M = sdpvar(m,m);

obj = trace(C*L*C') + trace(M); % Objective function

% LMI definitions 
lmi1 = A*L -B*X + L*A' - X'*B' + B*B' <= 0;
lmi2 = [M X; X' L] >= 0;
lmi3 = L >= 0;
lmi = [lmi1,lmi2,lmi3];

% Options
options = sdpsettings('solver','mosek');
optimize(lmi,obj,options);

% Controller
X = value(X);
L = value(L);
K_H2 = X*inv(L);

% Step response of closed loop system
Acl = A - B*K_H2;
Bcl = B;
Ccl = C;
Dcl = D;
sys_cl = ss(Acl,Bcl,Ccl,Dcl);

figure(21)
step(sys_cl)
title('Step response using K_{H2} controller')

% Step response with LQR
Q = C'*C;
R = eye(m);
[K_lqr,~,~] = lqr(A,B,Q,R);
Acl_lqr = A - B*K_lqr;
sys_cl_lqr = ss(Acl_lqr,Bcl,Ccl,Dcl);

% Comparison
figure(22)
step(sys_cl_lqr, '*')
hold on
step(sys_cl, '-g')
legend('Step response using LQR', 'Step response using H2')
title('Step response comparison')

%% Data-driven controller - multimodel
% Load empty structure
[SYS, OBJ, CON, PAR] = datadriven.utils.emptyStruct();

% ------------------------------------------------------------------------------
%   Initial controller
% ------------------------------------------------------------------------------
z = tf('z',Ts);
c = 0.001;
Kc = c / (1 - z^-1);                       % Initial controller
[num, den] = tfdata(Kc, 'v');   % Extract numerator and denominator

order = orderKreduced;
den(order + 1) = 0; % Zero padding to have same order as desired controller
num(order + 1) = 0; % Zero padding to have same order as desired controller

% Fixed parts of the controller
%   NOTE: Initial controller should contain the fixed parts too!
Fy = [1 -1];                 % Fixed part of denominator as polynomial
den = deconv(den, Fy);  % Remove fixed part of denominator

Fx = 1;                 % Fixed part of numerator as polynomial
num = deconv(num, Fx);  % Remove fixed part of numerator

SYS.controller.num = num;
SYS.controller.den = den;
SYS.controller.Ts = Ts;
SYS.controller.Fx = Fx;
SYS.controller.Fy = Fy;

% ------------------------------------------------------------------------------
%   Nominal system(s)
% ------------------------------------------------------------------------------
% Systems should be LTI systems (`ss`, `tf`, `frd`, ...)
SYS.model = Gmm;

% ------------------------------------------------------------------------------
%   Frequencies for controller synthesis
% ------------------------------------------------------------------------------
SYS.W = logspace(0,log10(pi/Ts),400);

% ------------------------------------------------------------------------------
%   Filters for objectives
% ------------------------------------------------------------------------------
% Filter should be LTI systems (`ss`, `tf`, `frd`, ...)
% For unused objectives, set filters to []
OBJ.oinf.W1 = W1;   % ║W1 S║∞
OBJ.oinf.W2 = [];   % ║W2 T║∞
OBJ.oinf.W3 = tf(W3);   % ║W3 U║∞
OBJ.oinf.W4 = [];   % ║W4 V║∞

OBJ.o2.W1 = [];     % ║W1 S║₂
OBJ.o2.W2 = [];     % ║W2 T║₂
OBJ.o2.W3 = [];     % ║W3 U║₂
OBJ.o2.W4 = [];     % ║W4 V║₂

OBJ.LSinf.Ld = [];  % ║W (Ld - G K)║∞
OBJ.LSinf.W  = [];

OBJ.LS2.Ld = [];    % ║W (Ld - G K)║₂
OBJ.LS2.W  = [];

% ------------------------------------------------------------------------------
%   Filters for constraints
% ------------------------------------------------------------------------------
% Filter should be LTI systems (`ss`, `tf`, `frd`, ...)
% For unused constraints, set filters to []
CON.W1 = [];    % ║W1 S║∞ ≤ 1
CON.W2 = [];    % ║W2 T║∞ ≤ 1
CON.W3 = [];    % ║W3 U║∞ ≤ 1
CON.W4 = [];    % ║W4 V║∞ ≤ 1

% ------------------------------------------------------------------------------
%   Optimisation parameters
% ------------------------------------------------------------------------------
PAR.tol = 1e-4;     % Numerical tolerance for convergence
PAR.maxIter = 100;  % Maximum number of allowed iterations

verbosity = true;   % To print controller synthesis iterations
solver = "mosek";        % Solver to use for optimisation ("mosek", "sedumi", ...)

% ------------------------------------------------------------------------------
%   Solve the datadriven controller synthesis problem
% ------------------------------------------------------------------------------
[K, sol] = datadriven.datadriven(SYS, OBJ, CON, PAR, verbosity, solver);

% Test performance
T = feedback(Gmm*K,1);
Tnom = feedback(Gnom*K,1);
U = feedback(K,Gmm);
Unom = feedback(K,Gnom);
S = feedback(1,Gmm*K);
Snom = feedback(1,Gnom*K);

figure(21)
subplot(2,2,1)
step(T,Tnom)
title('Step response')
legend('Multimodel','Nominal')
subplot(2,2,2)
step(U,Unom)
title('Control signal')
legend('Multimodel','Nominal')
ylim([-1.5,1.5])
subplot(2,2,3)
bodemag(U,Unom, 1/tf(W3))
xlim([10^-2 1560])
title('Sensitivity function U')
legend('Multimodel','Nominal','W3^-1')
subplot(2,2,4)
bodemag(S,Snom, W1^-1)
xlim([10^-2 1560])
title('Sensitivity function S')
legend('Multimodel','Nominal', 'W1 inv')
sgtitle('Datadriven - Multimodel')
saveas(gca,'CE2_datadriven_multimodel','png')

condition = norm(W1*S,inf);

if all(condition <= 1)
    fprintf('Robust performance conditions are met\n')
end
%%
Kred = reduce(K,8);

figure(23)
bodemag(K,Kred)
legend('Controller before reduction', 'Controller after reduction')
title('Reduction of the order of the controller')

% Test performance
T = feedback(Gmm*Kred,1);
Tnom = feedback(Gnom*Kred,1);
U = feedback(Kred,Gmm);
Unom = feedback(Kred,Gnom);
S = feedback(1,Gmm*Kred);
Snom = feedback(1,Gnom*Kred);

figure(24)
subplot(2,2,1)
step(T,Tnom)
title('Step response')
legend('Multimodel','Nominal')
subplot(2,2,2)
step(U,Unom)
title('Control signal')
legend('Multimodel','Nominal')
ylim([-1.5,1.5])
subplot(2,2,3)
bodemag(U,Unom, 1/tf(W3))
xlim([10^-2 1560])
title('Sensitivity function U')
legend('Multimodel','Nominal','W3^-1')
subplot(2,2,4)
bodemag(S,Snom, W1^-1)
xlim([10^-2 1560])
title('Sensitivity function S')
legend('Multimodel','Nominal', 'W1 inv')
sgtitle('Datadriven - Multimodel reduced')
saveas(gca,'CE2_datadriven_multimodel_reduced','png')

condition = norm(W1*S,inf)

if all(condition <= 1)
    fprintf('Robust performance conditions are met\n')
end
%% Data-driven controller - multiplicative
% Load empty structure
[SYS, OBJ, CON, PAR] = datadriven.utils.emptyStruct();

% ------------------------------------------------------------------------------
%   Initial controller
% ------------------------------------------------------------------------------
z = tf('z',Ts);
c = 0.001;
Kc = c / (1 - z^-1);                       % Initial controller
[num, den] = tfdata(Kc, 'v');   % Extract numerator and denominator

order = orderKreduced;
den(order + 1) = 0; % Zero padding to have same order as desired controller
num(order + 1) = 0; % Zero padding to have same order as desired controller

% Fixed parts of the controller
%   NOTE: Initial controller should contain the fixed parts too!
Fy = [1 -1];                 % Fixed part of denominator as polynomial
den = deconv(den, Fy);  % Remove fixed part of denominator

Fx = 1;                 % Fixed part of numerator as polynomial
num = deconv(num, Fx);  % Remove fixed part of numerator

SYS.controller.num = num;
SYS.controller.den = den;
SYS.controller.Ts = Ts;
SYS.controller.Fx = Fx;
SYS.controller.Fy = Fy;

% ------------------------------------------------------------------------------
%   Nominal system(s)
% ------------------------------------------------------------------------------
% Systems should be LTI systems (`ss`, `tf`, `frd`, ...)
SYS.model = Gnom;

% ------------------------------------------------------------------------------
%   Frequencies for controller synthesis
% ------------------------------------------------------------------------------
SYS.W = logspace(0,log10(pi/Ts),400);

% ------------------------------------------------------------------------------
%   Filters for objectives
% ------------------------------------------------------------------------------
% Filter should be LTI systems (`ss`, `tf`, `frd`, ...)
% For unused objectives, set filters to []
OBJ.oinf.W1 = W1;   % ║W1 S║∞
OBJ.oinf.W2 = W2;   % ║W2 T║∞
OBJ.oinf.W3 = tf(W3);   % ║W3 U║∞
OBJ.oinf.W4 = [];   % ║W4 V║∞

OBJ.o2.W1 = [];     % ║W1 S║₂
OBJ.o2.W2 = [];     % ║W2 T║₂
OBJ.o2.W3 = [];     % ║W3 U║₂
OBJ.o2.W4 = [];     % ║W4 V║₂

OBJ.LSinf.Ld = [];  % ║W (Ld - G K)║∞
OBJ.LSinf.W  = [];

OBJ.LS2.Ld = [];    % ║W (Ld - G K)║₂
OBJ.LS2.W  = [];

% ------------------------------------------------------------------------------
%   Filters for constraints
% ------------------------------------------------------------------------------
% Filter should be LTI systems (`ss`, `tf`, `frd`, ...)
% For unused constraints, set filters to []
CON.W1 = [];    % ║W1 S║∞ ≤ 1
CON.W2 = [];    % ║W2 T║∞ ≤ 1
CON.W3 = [];    % ║W3 U║∞ ≤ 1
CON.W4 = [];    % ║W4 V║∞ ≤ 1

% ------------------------------------------------------------------------------
%   Optimisation parameters
% ------------------------------------------------------------------------------
PAR.tol = 1e-4;     % Numerical tolerance for convergence
PAR.maxIter = 100;  % Maximum number of allowed iterations

verbosity = true;   % To print controller synthesis iterations
solver = "mosek";        % Solver to use for optimisation ("mosek", "sedumi", ...)

% ------------------------------------------------------------------------------
%   Solve the datadriven controller synthesis problem
% ------------------------------------------------------------------------------
[K, sol] = datadriven.datadriven(SYS, OBJ, CON, PAR, verbosity, solver);
K_mult = K;
% Test performance
T = feedback(Gmm*K,1);
Tnom = feedback(Gnom*K,1);
U = feedback(K,Gmm);
Unom = feedback(K,Gnom);
S = feedback(1,Gmm*K);
Snom = feedback(1,Gnom*K);

figure(22)
subplot(2,2,1)
step(T,Tnom)
title('Step response')
legend('Multimodel','Nominal')
subplot(2,2,2)
step(U,Unom)
title('Control signal')
legend('Multimodel','Nominal')
ylim([-1.5,1.5])
subplot(2,2,3)
bodemag(U,Unom, 1/tf(W3))
xlim([10^-2 1560])
title('Sensitivity function U')
legend('Multimodel','Nominal','W3^-1')
subplot(2,2,4)
bodemag(S,Snom, W1^-1)
xlim([10^-2 1560])
title('Sensitivity function S')
legend('Multimodel','Nominal', 'W1 inv')
sgtitle('Datadriven - Multiplicative')
saveas(gca,'CE2_datadriven_multiplicative','png')

condition1 = norm([W1*Snom W2*Tnom],inf);
condition2 = norm(W1*S,inf);

if condition1 <= 1/sqrt(2) && all(condition2 <= 1)
    fprintf('Robust performance conditions are met\n')
end

%%
Kred = reduce(K,8);
K_mult_red = Kred;
figure(23)
bodemag(K,Kred)
legend('Controller before reduction', 'Controller after reduction')
title('Reduction of the order of the controller')

% Test performance
T = feedback(Gmm*Kred,1);
Tnom = feedback(Gnom*Kred,1);
U = feedback(Kred,Gmm);
Unom = feedback(Kred,Gnom);
S = feedback(1,Gmm*Kred);
Snom = feedback(1,Gnom*Kred);

figure(25)
subplot(2,2,1)
step(T,Tnom)
title('Step response')
legend('Multimodel','Nominal')
subplot(2,2,2)
step(U,Unom)
title('Control signal')
legend('Multimodel','Nominal')
ylim([-1.5,1.5])
subplot(2,2,3)
bodemag(U,Unom, 1/tf(W3))
xlim([10^-2 1560])
title('Sensitivity function U')
legend('Multimodel','Nominal','W3^-1')
subplot(2,2,4)
bodemag(S,Snom, W1^-1)
xlim([10^-2 1560])
title('Sensitivity function S')
legend('Multimodel','Nominal', 'W1 inv')
sgtitle('Datadriven - Multiplicative reduced')
saveas(gca,'CE2_datadriven_multiplicative_reduced','png')

condition = norm(W1*S,inf)

if all(condition <= 1)
    fprintf('Robust performance conditions are met\n')
end

%%
X_mult = K.Numerator{1,1};
Y_mult = K.Denominator{1,1};
STR = join(compose('%d',typecast(Y_mult,'int64')),',') ; 
clipboard ('copy', STR{1})