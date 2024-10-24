function wNSB = NSB(doas, N)
    M = length(doas);

    % Compute the steering matrix A
    A = zeros(N, M);
    for i = 1:M
        A(:, i) = steering_vector(doas(i), N);
    end

    % Calculate Hermitian transpose of A
    AH = A';

    % AH and A multiplication and diagonal loading
    AHA = AH * A + 1e-6 * eye(M);

    % Unit vector
    u1 = zeros(M, 1);
    u1(1) = 1;

    % Final steps
    wNSB = A * (AHA \ u1);
end