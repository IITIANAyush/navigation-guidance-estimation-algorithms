clear; clc; close all;

dt       = 0.1;
max_time = 200;

% 5 different combinations: [Vm, Vt, theta, delta, alpha_T]
param_sets = [ ...
    300, 200, deg2rad(0),   deg2rad(10),  deg2rad(0);   % Case 1 (same as Python example)
    300, 200, deg2rad(20),  deg2rad(10),  deg2rad(0);   % Case 2: target initially offset
    300, 200, deg2rad(0),   deg2rad(5),   deg2rad(0);   % Case 3: smaller deviation angle
    300, 200, deg2rad(0),   deg2rad(15),  deg2rad(0);   % Case 4: larger deviation angle
    300, 200, deg2rad(0),   deg2rad(10),  deg2rad(20)]; % Case 5: target moving off-boresight

num_cases = size(param_sets, 1);

% Store outputs if you want later
missile_trajs   = cell(num_cases, 1);
target_trajs    = cell(num_cases, 1);
times_list      = cell(num_cases, 1);
latacc_list     = cell(num_cases, 1);

for i = 1:num_cases
    params = param_sets(i, :);
    [m_traj, t_traj, t_vec, a_lat] = simulate_deviated_pursuit(params, dt, max_time);

    missile_trajs{i} = m_traj;
    target_trajs{i}  = t_traj;
    times_list{i}    = t_vec;
    latacc_list{i}   = a_lat;
end

%% Plot trajectories for all 5 cases
figure;
for i = 1:num_cases
    subplot(2, 3, i);
    m_traj = missile_trajs{i};
    t_traj = target_trajs{i};

    plot(m_traj(:,1), m_traj(:,2), 'LineWidth', 1.2); hold on;
    plot(t_traj(:,1), t_traj(:,2), '--', 'LineWidth', 1.0);
    grid on; axis equal;
    xlabel('X (m)');
    ylabel('Y (m)');
    title(sprintf('Trajectories - Case %d', i));
    if i == 1
        legend('Missile', 'Target');
    end
end

%% Plot lateral acceleration profiles for all 5 cases
figure;
for i = 1:num_cases
    subplot(2, 3, i);
    t_vec = times_list{i};
    a_lat = latacc_list{i};

    plot(t_vec, a_lat, 'LineWidth', 1.2);
    grid on;
    xlabel('Time (s)');
    ylabel('a_{lat} (m/s^2)');
    title(sprintf('Lateral Accel - Case %d', i));
end
