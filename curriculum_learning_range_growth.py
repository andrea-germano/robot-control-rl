import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE DATI ---
TOTAL_TIMESTEPS = 2_000_000
STEPS_RESOLUTION = 1000

# Generazione asse X
timesteps = np.linspace(0, TOTAL_TIMESTEPS, STEPS_RESOLUTION)
progress = timesteps / TOTAL_TIMESTEPS

# --- DEFINIZIONE CURVE ---
# 1. Lineare (Red)
# Range: 15% -> 45%
lin_min, lin_max = 15, 45  
y_linear = lin_min + (lin_max - lin_min) * progress

# 2. Quadratica Piccolo (Black)
# Range: 15% -> 45%
quad1_min, quad1_max = 15, 45
y_quad_small = quad1_min + (quad1_max - quad1_min) * (progress ** 2)

# 3. Quadratica Grande (Violet)
# Range: 15% -> 60%
quad2_min, quad2_max = 15, 60
y_quad_large = quad2_min + (quad2_max - quad2_min) * (progress ** 2)

# --- SOGLIA TARGET ---
# ~39.5% (punto in cui si copre il gap di 1kg)
TARGET_COVERAGE_PCT = 39.5 

# --- PLOTTING ---
plt.figure(figsize=(8, 5))
# Imposta font serif (Times New Roman style) per coerenza con IEEE
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
plt.grid(True, linestyle='--', alpha=0.4)

# 1. Plot Lineare (Red) - Tratteggiata per indicare che è una baseline meno efficace
plt.plot(timesteps, y_linear, 
         label=f'Linear ({lin_min}-{lin_max}%)', 
         color='red', linestyle='--', linewidth=2)

# 2. Plot Quadratico Piccolo (Black) - Solida
plt.plot(timesteps, y_quad_small, 
         label=f'Quadratic Small ({quad1_min}-{quad1_max}%)', 
         color='black', linewidth=2.5)

# 3. Plot Quadratico Grande (Violet) - Solida
plt.plot(timesteps, y_quad_large, 
         label=f'Quadratic Wider ({quad2_min}-{quad2_max}%)', 
         color='violet', linewidth=2.5) # Usa 'darkviolet' se 'violet' risulta troppo chiaro stampato

# --- LINEA ORIZZONTALE TARGET ---
plt.axhline(y=TARGET_COVERAGE_PCT, color='gray', linestyle='-.', linewidth=1.5, alpha=0.8)
plt.text(TOTAL_TIMESTEPS * 0.02, TARGET_COVERAGE_PCT + 1.5, 
         'Target Mass Coverage (~39.5%)', 
         fontsize=10, color='gray', style='italic', weight='bold')

# --- FORMATTAZIONE ASSI ---
plt.xlabel('Training Timesteps', fontsize=12)
# Formatta X in Milioni (0.0M, 0.5M, ...)
ticks_x = np.linspace(0, TOTAL_TIMESTEPS, 5)
plt.xticks(ticks_x, [f'{x/1e6:.1f}M' for x in ticks_x])
plt.xlim(0, TOTAL_TIMESTEPS)

plt.ylabel('Randomization Range ($\pm \%$)', fontsize=12)
plt.ylim(0, 65)

# Titolo e Legenda
plt.title('Curriculum Learning Schedules Comparison', fontsize=14, pad=10)
plt.legend(loc='lower right', frameon=True, fancybox=True, framealpha=0.9)

plt.tight_layout()

# Salva immagine
filename = 'curriculum_schedules.png'
plt.savefig(filename, dpi=300)
print(f"Saved as {filename}")

plt.show()