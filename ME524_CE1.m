%% Computer exercise 1
clc, clear, close all

%% 1.1
fprintf('SISO\n')
G = tf([1 -1],[1 2 10]); % Transfer function

% 1.1.1
% Frequency response approximative
omega = 0:0.01:10000;  % Range of frequencies
Gjw = freqresp(G, omega);
dw = omega(2)-omega(1);
norm2_freq = sqrt((1/pi)*(sum((abs(Gjw).^2).*dw)));

% Impulse response
syms s
Numerator = poly2sym(G.Numerator{1,1},s);
Denominator = poly2sym(G.Denominator{1,1},s);
Gsym = Numerator/Denominator;
g = matlabFunction(ilaplace(Gsym)); % Inverse Laplace transform
norm2_impulse = sqrt(integral(@(t) (abs(g(t)).^2),0,Inf));

% State-space 
[A,B,C,D] = ssdata(G); % State-space matrices
L = are(A',zeros(2,2),B*B');
norm2_ss = sqrt(trace(C*L*C'));

% True value
true_norm2 = norm(G,2);

fprintf(['H_2: \n' ...
    'Frequency response => %.4f\n' ...
    'Impulse response => %.4f\n' ...
    'State-space method => %.4f\n' ...
    'True value => %.4f\n\n'], norm2_freq, norm2_impulse, norm2_ss, true_norm2)

% 1.1.2
% Frequency response
omega = 0:0.001:100;  % Range of frequencies
Gjw = freqresp(G, omega);
normInf_freq = max(abs(Gjw));

% Bounded real lemma
gu = 1;
gl = 0.1;

eps = 1e-8;

while (gu-gl)/gl > eps
    g = (gu+gl)/2;
    
    H = [A g^(-2)*(B*B') ; -C'*C -A']; % Hamiltonian matrix

    if any(abs(real(eig(H))) < 1e-5) % Eigenvalue on the imaginary axis
        gl = g;
    else % Eigenvalues not on the imaginary axis
        gu = g;
    end
end

normInf_lemma = (gu+gl)/2;

% True value
true_normInf = norm(G,inf);

fprintf(['H_Inf: \n' ...
    'Frequency response => %.4f\n' ...
    'Impulse response => %.4f\n' ...
    'True value => %.4f\n\n'], normInf_freq, normInf_lemma, true_normInf)
%% 1.2
fprintf('MIMO\n')

A = [20 -27 7 ; 53 -63 13 ; -5 12 -8];
B = [1 -1 ; -2 -1 ; -3 0];
C = [0 0 -2 ; 1 -1 -1];
D = zeros(2,2);

sys = ss(A,B,C,D);

G = tf(sys);

% 1.2.1
% Frequency response 
omega = logspace(-4,5,1000);  % Range of frequencies
Gjw = [];
for i = 1:size(omega,2)
    Gjw = [Gjw, trace(freqresp(conj(G)'*G, omega(i)))];
end
norm2_freq = abs(sqrt(1/pi*trapz(omega, Gjw)));

% State-space
L = are(A',zeros(3,3),B*B');
norm2_ss = sqrt(trace(C*L*C'));

% True value
true_norm2 = norm(G,2);

fprintf(['H_2: \n' ...
    'Frequency response => %.4f\n' ...
    'State-space method => %.4f\n' ...
    'True value => %.4f\n\n'], norm2_freq, norm2_ss, true_norm2)

% 1.2.2
% Frequency response
omega = logspace(-4,5,1000);  % Range of frequencies
Gjw = freqresp(conj(G)'*G, omega);
eigen = [];
for i = 1:size(omega,2)
    eigen = [eigen, real(eig(freqresp(conj(G)'*G,omega(i))))];
end
normInf_freq = max(sqrt(max(eigen)));

% Bounded real lemma
gu = 3;
gl = 0.1;

eps = 1e-8;

while (gu-gl)/gl > eps
    g = (gu+gl)/2;
    
    H = [A g^(-2)*(B*B') ; -C'*C -A']; % Hamiltonian matrix

    if any(abs(real(eig(H))) < 1e-5) % Eigenvalue on the imaginary axis
        gl = g;
    else % Eigenvalues not on the imaginary axis
        gu = g;
    end
end

normInf_lemma = (gu+gl)/2;

% True value
true_normInf = norm(G,Inf);

fprintf(['H_Inf: \n' ...
    'Frequency response => %.4f\n' ...
    'Impulse response => %.4f\n' ...
    'True value => %.4f\n\n'], normInf_freq, normInf_lemma, true_normInf)

%% 1.3
% Uncertain model
a = ureal('a',11,'PlusMinus',1);
b = ureal('b',4,'PlusMinus',1);
c = ureal('c',9,'PlusMinus',2);
G = tf(a,[1 b c]);

nominalG = G.NominalValue;

% Bode, Nyquist and step response
figure(1)
subplot(1,3,1)
step(G)
hold on
step(nominalG)
legend('Uncertain System','Nominal System')
subplot(1,3,2)
bode(G)
hold on
bode(nominalG)
legend('Uncertain System','Nominal System')
subplot(1,3,3)
nyquist(G)
hold on
nyquist(nominalG)
legend('Uncertain System','Nominal System')

% Multimodel uncertainty
usys20 = usample(G,20);
usys200 = usample(G,200);


% Convert to multiplicative uncertainty
[~,Info_20] = ucover(usys20, nominalG, 1); % filter of order 1
[~,Info_200] = ucover(usys200, nominalG, 1);
[~,Info2_20] = ucover(usys20, nominalG, 2); % filter of order 2
[~,Info2_200] = ucover(usys200, nominalG, 2);
[~,Info3_20] = ucover(usys20, nominalG, 3); % filter of order 3
[~,Info3_200] = ucover(usys200, nominalG, 3);
[~,Info4_20] = ucover(usys20, nominalG, 4); % filter of order 4
[~,Info4_200] = ucover(usys200, nominalG, 4);

figure(3); % Plot for 20 samples
hold on
bodemag(usys20/nominalG -1);
bodemag(Info_20.W1, '-g');% filter of order 1
bodemag(Info2_20.W1, '-r'); % filter of order 2
bodemag(Info3_20.W1, '-b'); % filter of order 3
bodemag(Info4_20.W1, '-y'); % filter of order 4
xlim([10^(-2) 10^2])
legend('Magnitude deviation of uncertain system', '1st order filter', '2nd order filter', '3rd order filter', '4th order filter', 'fontsize', 14, 'Location', 'SouthWest')
grid on

figure(4); % Plot for 200 samples
hold on
bodemag(usys200/nominalG -1);
bodemag(Info_200.W1, '-g'); % filter of order 1
bodemag(Info2_200.W1, '-r'); % filter of order 2
bodemag(Info3_200.W1, '-b'); % filter of order 3
bodemag(Info4_200.W1, '-y'); % filter of order 4
xlim([10^(-2) 10^2])
legend('Magnitude deviation of uncertain system', '1st order filter', '2nd order filter', '3rd order filter', '4th order filter', 'fontsize', 14, 'Location', 'SouthWest')
grid on

figure(5)
hold on
bodemag(ss(Info_20.W1.A,Info_20.W1.B,Info_20.W1.C,Info_20.W1.D))
bodemag(ss(Info2_20.W1.A,Info2_20.W1.B,Info2_20.W1.C,Info2_20.W1.D))
bodemag(ss(Info3_20.W1.A,Info3_20.W1.B,Info3_20.W1.C,Info3_20.W1.D))
bodemag(ss(Info4_20.W1.A,Info4_20.W1.B,Info4_20.W1.C,Info4_20.W1.D))
xlim([10^(-2) 10^2])
legend( '1st order filter', '2nd order filter', '3rd order filter', '4th order filter', 'fontsize', 14, 'Location', 'SouthWest')
title('20 Samples')
grid on

figure(6)
hold on
bodemag(ss(Info_200.W1.A,Info_200.W1.B,Info_200.W1.C,Info_200.W1.D))
bodemag(ss(Info2_200.W1.A,Info2_200.W1.B,Info2_200.W1.C,Info2_200.W1.D))
bodemag(ss(Info3_200.W1.A,Info3_200.W1.B,Info3_200.W1.C,Info3_200.W1.D))
bodemag(ss(Info4_200.W1.A,Info4_200.W1.B,Info4_200.W1.C,Info4_200.W1.D))
xlim([10^(-2) 10^2])
legend( '1st order filter', '2nd order filter', '3rd order filter', '4th order filter', 'fontsize', 14, 'Location', 'SouthWest')
title('200 Samples')
grid on