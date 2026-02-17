import numpy as np
import matplotlib.pyplot as plt

def simulate_deviated_pursuit(params, dt=0.1, max_time=400):
    Vm, Vt, theta0, delta, alpha_T = params

    # Initial separation (50 km)
    R0 = 50000.0
    missile_pos = np.array([0.0, 0.0], dtype=float)
    target_pos  = np.array([R0 * np.cos(theta0), R0 * np.sin(theta0)], dtype=float)

    # Target velocity vector
    Vt_vec = Vt * np.array([np.cos(alpha_T), np.sin(alpha_T)])

    eps = 1e-8

    missile_traj = [missile_pos.copy()]
    target_traj  = [target_pos.copy()]
    times        = [0.0]
    a_lat_list   = [0.0]

    t = 0.0

    while np.linalg.norm(target_pos - missile_pos) > 10.0 and t < max_time:
        # Geometry
        los_vec = target_pos - missile_pos
        R       = np.linalg.norm(los_vec)
        theta   = np.arctan2(los_vec[1], los_vec[0])   # LOS angle

        # Missile heading (deviated pursuit)
        alpha_M = theta + delta
        Vm_vec  = Vm * np.array([np.cos(alpha_M), np.sin(alpha_M)])

        #  Lateral acceleration from lecture engagement equations 
        if R > eps:
            # R * theta_dot = Vt * sin(alpha_T - theta) - Vm * sin(delta)
            theta_dot = (Vt * np.sin(alpha_T - theta) - Vm * np.sin(delta)) / R
            a_lat = Vm * theta_dot
        else:
            theta_dot = 0.0
            a_lat = 0.0

        # Update positions
        missile_pos += Vm_vec * dt
        target_pos  += Vt_vec * dt
        t += dt

        # Store
        missile_traj.append(missile_pos.copy())
        target_traj.append(target_pos.copy())
        times.append(t)
        a_lat_list.append(a_lat)

    return (np.array(missile_traj),
            np.array(target_traj),
            np.array(times),
            np.array(a_lat_list))


# Different Scenarios from project table 
scenarios = [
    # [Vm, Vt, θ0, δ, αT]
    [800, 400, np.deg2rad(30), np.deg2rad(10), np.deg2rad(45)],  # Scenario 1
    [700, 400, np.deg2rad(30), np.deg2rad(10), np.deg2rad(45)],  # Scenario 2
    [600, 400, np.deg2rad(30), np.deg2rad(10), np.deg2rad(45)],  # Scenario 3
    [600, 400, np.deg2rad(30), np.deg2rad(30), np.deg2rad(45)],  # Scenario 4
    [600, 300, np.deg2rad(30), np.deg2rad(30), np.deg2rad(45)],  # Scenario 5
]

results = [simulate_deviated_pursuit(p) for p in scenarios]

#  PLOT TRAJECTORIES

plt.figure(figsize=(15, 10))
for i, (m, t, time, a_lat) in enumerate(results):
    plt.subplot(2, 3, i + 1)
    plt.plot(m[:, 0] / 1000, m[:, 1] / 1000, label='Missile')
    plt.plot(t[:, 0] / 1000, t[:, 1] / 1000, '--', label='Target')
    plt.title(f"Trajectories – Scenario {i+1}")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.grid(True)
    plt.axis("equal")
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.show()


#  PLOT LATERAL ACCELERATIONS




plt.figure(figsize=(15, 10))
for i, (m, t, time, a_lat) in enumerate(results):
    plt.subplot(2, 3, i + 1)
    plt.plot(time, a_lat)
    plt.title(f"Lateral Acceleration – Scenario {i+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("a$_M$ (m/s²)")
    plt.grid(True)

plt.tight_layout()
plt.show()




#  PLOT POSITION vs TIME  (Missile & Target)

plt.figure(figsize=(15, 10))
for i, (m, t, time, a_lat) in enumerate(results):
    plt.subplot(2, 3, i + 1)
    plt.plot(time, m[:, 0] / 1000, label='Missile X (km)')
    plt.plot(time, m[:, 1] / 1000, label='Missile Y (km)')
    plt.plot(time, t[:, 0] / 1000, '--', label='Target X (km)')
    plt.plot(time, t[:, 1] / 1000, '--', label='Target Y (km)')
    
    plt.title(f"Position vs Time – Scenario {i+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (km)")
    plt.grid(True)
    
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.show()

