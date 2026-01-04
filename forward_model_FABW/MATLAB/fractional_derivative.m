function D = fractional_derivative(x, alpha, dt)
%FRACTIONAL_DERIVATIVE
% Grünwald–Letnikov fractional derivative (stable recursive form)
%
% D^alpha x(t_i) ≈ (1/dt^alpha) * sum_{j=0}^i c_j * x(i-j)
% c_0 = 1
% c_j = c_{j-1} * ((j-1 - alpha)/j)

x = x(:);
N = length(x);
D = zeros(N,1);

c = zeros(N,1);
c(1) = 1.0;

for j = 2:N
    c(j) = c(j-1) * ((j-2 - alpha)/(j-1));
end

inv_dt_alpha = dt^(-alpha);

for i = 2:N
    xi = x(i:-1:1);
    ci = c(1:i);
    D(i) = inv_dt_alpha * (ci.' * xi);
end
end
