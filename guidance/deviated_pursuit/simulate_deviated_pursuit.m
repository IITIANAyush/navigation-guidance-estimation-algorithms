function [missile_traj, target_traj, times, lateral_accels] = simulate_deviated_pursuit(params, dt, max_time)
% params = [Vm, Vt, theta, delta, alpha_T]

Vm      = params(1);
Vt      = params(2);
theta   = params(3);
delta   = params(4);
alpha_T = params(5);

% Initial positions
R0 = 50000;  % 50 km initial separation
missile_pos = [0, 0];
target_pos  = [R0*cos(theta), R0*sin(theta)];

% Target velocity (constant heading)
Vt_vec = Vt * [cos(alpha_T), sin(alpha_T)];

% Initialize logs
missile_traj    = missile_pos;
target_traj     = target_pos;
times           = 0;
lateral_accels  = 0;

t   = 0;
eps = 1e-6;

while norm(target_pos - missile_pos) > 10 && t < max_time
    % LOS geometry
    los_vec   = target_pos - missile_pos;
    los_angle = atan2(los_vec(2), los_vec(1));

    % Missile velocity (deviated pursuit)
    alpha_M = los_angle + delta;
    Vm_vec  = Vm * [cos(alpha_M), sin(alpha_M)];

    % Lateral acceleration (simple centripetal model)
    R_los = norm(los_vec);
    if abs(delta) > eps
        turn_radius = R_los / (2 * sin(delta));
        a_lat = Vm^2 / turn_radius;
    else
        a_lat = 0;
    end

    % Update states
    missile_pos = missile_pos + Vm_vec * dt;
    target_pos  = target_pos  + Vt_vec * dt;

    % Update time and logs
    t = t + dt;
    missile_traj   = [missile_traj; missile_pos];   %#ok<AGROW>
    target_traj    = [target_traj; target_pos];     %#ok<AGROW>
    times          = [times; t];                    %#ok<AGROW>
    lateral_accels = [lateral_accels; a_lat];       %#ok<AGROW>
end
end
