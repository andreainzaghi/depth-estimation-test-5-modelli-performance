import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_relative_error(true_csv, pred_csv):
    # Carica i due CSV
    true_data = pd.read_csv(true_csv)
    pred_data = pd.read_csv(pred_csv)
    
    # Unisci i due dataset sui pixel corrispondenti (u, v)
    merged_data = pd.merge(true_data, pred_data, on=['u', 'v'], suffixes=('_true', '_pred'))
    
    # Estrai le profondità
    z_true = merged_data['z_true'].values
    z_pred = merged_data['z_pred'].values
    
    # Calcola il fattore di scala
    scale_factor = np.median(z_true) / np.median(z_pred)
    
    # Trasforma le profondità predette in valori assoluti
    z_pred_absolute = z_pred * scale_factor
    
    # Calcola l'errore relativo
    relative_error = np.abs(z_pred_absolute - z_true) / z_true
    
    # Calcola l'errore relativo medio
    mean_relative_error = np.mean(relative_error)
    print(f"Errore relativo medio: {mean_relative_error:.4f}")
    print(f"Fattore di scala calcolato: {scale_factor:.4f}")
    
    # Plotta il grafico
    plt.scatter(z_true, relative_error, alpha=0.5, s=10)
    plt.xlabel("Profondità vera (z_true)")
    plt.ylabel("Errore relativo")
    plt.title("Errore relativo vs Profondità")
    plt.grid(True)
    plt.show()
    
    return scale_factor, mean_relative_error

# Percorsi dei file CSV
true_csv = "img/disparity_kitty_one_uvz.csv"  # CSV con le profondità assolute
pred_csv = "output/depth_map/kitty_one_pred_depth.csv"  # CSV con le profondità relative

# Calcola l'errore relativo e il fattore di scala
scale_factor, mean_relative_error = compute_relative_error(true_csv, pred_csv)
