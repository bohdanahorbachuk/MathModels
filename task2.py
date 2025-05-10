import numpy as np
import matplotlib.pyplot as plt

def compute_concentration(Q, U, h, v, t_vert,
                          x_min=1.0, x_max=8000.0, y_min=-60.0, y_max=60.0,
                          nx=600, ny=600):
    H = h + v * t_vert  # Ефективна висота
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)

    r = 2
    s = 1
    q = 0.08
    p = 0.005
    
    sigma_y = q * X
    sigma_z = np.sqrt(2) * p * X
    A = Q / np.sqrt(2 * np.pi) * 1 / (p * q * X**2 * U)

    C = A * np.exp(-(Y / (np.sqrt(2 * sigma_y)))**r - (H / (np.sqrt(2 * sigma_z)))**s)
    return X, Y, C

def compute_metrics(Q, U, h, v, t_vert,
                    x_min=1.0, x_max=8000.0, y_min=-60.0, y_max=60.0,
                          nx=600, ny=600, threshold_fraction=0.05):
    X, Y, C = compute_concentration(Q, U, h, v, t_vert, x_min, x_max, y_min, y_max, nx, ny)
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    C_max = np.max(C)
    threshold = threshold_fraction * C_max
    area = np.sum(C > threshold) * dx * dy
    max_idx = np.unravel_index(np.argmax(C), C.shape)
    max_location = (X[max_idx], Y[max_idx])
    return area, max_location, C_max


# Базові значення параметрів
Q_base = 2.0      
U_base = 15.0    
h_base = 20.0    
v_base = 4.0    
t_vert_base = 2.0

# Діапазони параметрів
Q_range = np.linspace(1.0, 5.0, 10)
#U_range = np.linspace(10.0, 20.0, 5)
U_range = np.linspace(10.0, 20.0, 10)
h_range = np.linspace(15.0, 25.0, 10)
v_range = np.linspace(2.0, 6.0, 10)
t_vert_range = np.linspace(1.0, 3.0, 10)


# Словник збереження результатів
results = {'Q': {'param': Q_range, 'area': [], 'max_x': [], 'C_max': []},
           'U': {'param': U_range, 'area': [], 'max_x': [], 'C_max': []},
           'h': {'param': h_range, 'area': [], 'max_x': [], 'C_max': []},
           'v': {'param': v_range, 'area': [], 'max_x': [], 'C_max': []},
           't_vert': {'param': t_vert_range, 'area': [], 'max_x': [], 'C_max': []}}

# Додавання результатів у словник
def add_result(param_name, param_value, area, max_loc, C_max):
    results[param_name]['area'].append(area)
    results[param_name]['max_x'].append(max_loc[0])
    results[param_name]['C_max'].append(C_max)


# Обчислення метрик та вивід результатів у консоль
for param_name, param_range in results.items():
    print(f"\n### Analysis of the parameter: {param_name} ###")
    for param_value in param_range['param']:
        if param_name == 'Q':
            area, max_loc, C_max = compute_metrics(Q=param_value, U=U_base, h=h_base, v=v_base, t_vert=t_vert_base)
        elif param_name == 'U':
            area, max_loc, C_max = compute_metrics(Q=Q_base, U=param_value, h=h_base, v=v_base, t_vert=t_vert_base)
        elif param_name == 'h':
            area, max_loc, C_max = compute_metrics(Q=Q_base, U=U_base, h=param_value, v=v_base, t_vert=t_vert_base)
        elif param_name == 'v':
            area, max_loc, C_max = compute_metrics(Q=Q_base, U=U_base, h=h_base, v=param_value, t_vert=t_vert_base)
        elif param_name == 't_vert':
            area, max_loc, C_max = compute_metrics(Q=Q_base, U=U_base, h=h_base, v=v_base, t_vert=param_value)

        add_result(param_name, param_value, area, max_loc, C_max)

        print(f"{param_name} = {param_value:.2f} | Area = {area:.2f} м² | Max. concentration = {C_max:.2e} | Coordinates of the maximum = ({max_loc[0]:.2f}, {max_loc[1]:.2f})")

# Збереження окремих графіків для параметрів Q і U
for pname in ['Q', 'U']:
    p_values = results[pname]['param']
    
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    axes[0].plot(p_values, results[pname]['area'], 'o-', color='blue')
    axes[0].set_xlabel(pname)
    axes[0].set_ylabel("Pollution area (m²)")
    axes[0].set_title(f"Area vs {pname}")

    axes[1].plot(p_values, results[pname]['max_x'], 's-', color='red')
    axes[1].set_xlabel(pname)
    axes[1].set_ylabel("X-coordinate of the maximum (m)")
    axes[1].set_title(f"X-coordinate vs {pname}")

    axes[2].plot(p_values, results[pname]['C_max'], 'd-', color='green')
    axes[2].set_xlabel(pname)
    axes[2].set_ylabel("Max. concentration")
    axes[2].set_title(f"Max. concentration vs {pname}")

    plt.tight_layout()
    plt.savefig(f'plot_{pname}.png', dpi=300)
    plt.close(fig)



# Об'єднання графіків для h, v, t_vert в один файл
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
param_names = ['h', 'v', 't_vert']
titles = ["Pollution area", "X-coordinate of the maximum", "Max. concentration"]

for i, pname in enumerate(param_names):
    p_values = results[pname]['param']

    axes[0, i].plot(p_values, results[pname]['area'], 'o-', color='blue')
    axes[0, i].set_xlabel(pname)
    axes[0, i].set_ylabel("Area (m²)")
    axes[0, i].set_title(f"{titles[0]} vs {pname}")

    axes[1, i].plot(p_values, results[pname]['max_x'], 's-', color='red')
    axes[1, i].set_xlabel(pname)
    axes[1, i].set_ylabel("X-coordinate (m)")
    axes[1, i].set_title(f"{titles[1]} vs {pname}")

    axes[2, i].plot(p_values, results[pname]['C_max'], 'd-', color='green')
    axes[2, i].set_xlabel(pname)
    axes[2, i].set_ylabel("C_max")
    axes[2, i].set_title(f"{titles[2]} vs {pname}")

plt.tight_layout()
plt.savefig('plot_h_v_tvert.png', dpi=300)
plt.close(fig)



def compute_metrics_exact(Q, U, h, v, t_vert,
                    x_min=1.0, x_max=8000.0, y_min=-60.0, y_max=60.0,
                          nx=600, ny=600, threshold_fraction=0.1):
    X, Y, C = compute_concentration(Q, U, h, v, t_vert, x_min, x_max, y_min, y_max, nx, ny)
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    C_max = np.max(C)
    threshold = threshold_fraction * C_max
    area = np.sum(C > threshold) * dx * dy
    max_idx = np.unravel_index(np.argmax(C), C.shape)
    max_location = (X[max_idx], Y[max_idx])
    return area, max_location, C_max, X, Y, C


# Візуалізація розподілу концентрації для вибраної комбінації параметрів
Q_selected = 2.0
U_selected = 15.0
h_selected = 20.0
v_selected = 4.0
t_vert_selected = 2.0

_, _, _, X, Y, C = compute_metrics_exact(Q_selected, U_selected, h_selected, v_selected, t_vert_selected)

plt.figure(figsize=(8, 6))
plt.contourf(X, Y, C, levels=50, cmap='viridis')
plt.colorbar(label="Concentration")
plt.xlabel('Distance in the wind direction x (m)')
plt.ylabel('Distance perpendicular to the wind y (m)')
plt.title(f'Сoncentration distribution', fontsize=10)

plt.savefig('concentration_distribution.png', dpi=300)
plt.close()
