function [R, S] = poleplace(B, A, Hr, Hs, P)
    % Perform Pole placement
    % INPUTS:
    % A = [1 a1 a2...]
    % B = [b0 b1 b2...]
    % Hr = [hr0 hr1 ...]
    % Hs = [hs0 hs1 ...]
    % P = [1 p1 p2 ...]
    % OUTPUTS:
    % R = [r0 r1 r2...]
    % S = [1 s1 s2 ...]
    na = length(A) - 1;
    nb = length(B) - 1;
    nHs = length(Hs) - 1;
    nHr = length(Hr) - 1;
    np = length(P) - 1;

    if np > na + nHs + nb + nHr - 1
        error('Dimensions do not match!')
    end
    
    if size(A,2) ~= 1
        A = A';
    end

    if size(B,2) ~= 1
        B = B';
    end
    if size(Hs,2) ~= 1
        Hs = Hs';
    end
    if size(Hr,2) ~= 1
        Hr = Hr';
    end
    if size(P,2) ~= 1
        P = P';
    end

    na_prime = na + nHs;
    nb_prime = nb + nHr;

    A_prime = conv(A,Hs);
    B_prime = conv(B,Hr);
    
    for i = 1:nb_prime
        M1(:,i) = [zeros(i-1,1) ; A_prime ; zeros(nb_prime-i,1)];
    end
    for j = 1:na_prime
        M2(:,j) = [zeros(j-1,1) ; B_prime ; zeros(na_prime-j,1)];
    end

    M = [M1 M2];
    
    if np < na_prime+nb_prime
        P = [P ; zeros(na_prime+nb_prime-np-1,1)];
    end
    x = inv(M)*P;

    nr_prime = na_prime-1;
    ns_prime = nb_prime -1;

    S_prime = x(1:ns_prime+1); 
    R_prime = x(ns_prime+2:end);
    
    S = conv(Hs,S_prime);
    R = conv(Hr,R_prime);
end