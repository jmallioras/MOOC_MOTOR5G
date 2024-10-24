function sv = steering_vector(theta, N)
    % Simplified steering vector for angle theta considering a ULA of N elements,
    % isotropic point sources, and the distance between elements: lambda/2
    sv = exp(-1j * (0:N-1).' * pi * cosd(theta));
end