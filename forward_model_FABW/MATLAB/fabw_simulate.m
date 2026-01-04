function out = fabw_simulate(t, i, params)
%FABW_SIMULATE
% Fractional Asymmetric Bouc–Wen forward model (MATLAB)
% Strictly aligned with Python FABW_model.py

t = t(:);
i = i(:);
N = length(t);
dt = t(2) - t(1);

%% Current derivatives
di_dt = gradient(i, dt);
Di_frac = fractional_derivative(i, params.alpha_frac, dt);

%% State initialization
z = zeros(N,1);
F = zeros(N,1);

for k = 2:N
    z_prev = z(k-1);
    z_abs  = abs(z_prev);

    %% Bouc–Wen core
    core = ...
        params.A * di_dt(k) ...
      - params.beta  * abs(di_dt(k)) * (z_abs^(params.n-1)) * z_prev ...
      - params.gamma * di_dt(k)      * (z_abs^params.n);

    %% Fractional memory term (on current)
    frac_term = params.frac_gain * Di_frac(k);

    dz_dt_raw = core + frac_term;

    %% Asymmetric correction
    asym_term = params.delta_asym ...
              * tanh(params.tanh_gain * z_prev) ...
              * abs(dz_dt_raw);

    dz_dt = dz_dt_raw + asym_term;

    %% Euler integration
    z(k) = z_prev + dz_dt * dt;

    %% Output force
    F(k) = ...
        params.k1 * i(k) ...
      + params.k2 * di_dt(k) ...
      + params.alpha_z * z(k);
end

%% Output
out.t = t;
out.i = i;
out.z = z;
out.di_dt = di_dt;
out.Di_frac = Di_frac;
out.F = F;
end
