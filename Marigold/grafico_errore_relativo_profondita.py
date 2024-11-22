import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_relative_error(true_csv, pred_csv, output_plot="relative_error_plot.png"):
    """
    Calcola l'errore relativo tra due file CSV con colonne 'u', 'v', 'z',
    ordina i risultati e genera un grafico, ignorando i pixel mancanti.

    Args:
        true_csv (str): Percorso al CSV con le profondità vere.
        pred_csv (str): Percorso al CSV con le profondità predette.
        output_plot (str): Percorso per salvare il grafico (opzionale).
    """
    # Carica i dati
    true_data = pd.read_csv(true_csv)
    pred_data = pd.read_csv(pred_csv)

    # Unisci i due dataset sui pixel corrispondenti (u, v), ignorando quelli non presenti in entrambi
    merged_data = pd.merge(true_data, pred_data, on=['u', 'v'], suffixes=('_true', '_pred'), how='inner')

    # Controlla se ci sono dati da elaborare
    if merged_data.empty:
        print("Nessun pixel corrispondente trovato tra i due file!")
        return

    # Estrai profondità
    z_true = merged_data['z_true'].values
    z_pred = merged_data['z_pred'].values

    # Calcola l'errore relativo
    relative_error = np.abs(z_pred - z_true) / z_true

    # Combina profondità e errore in una matrice
    error_depth = np.column_stack((z_true, relative_error))

    # Ordina per profondità crescente
    error_depth_sorted = error_depth[error_depth[:, 0].argsort()]

    # Estrai profondità e errore relativi ordinati
    z_sorted = error_depth_sorted[:, 0]
    error_sorted = error_depth_sorted[:, 1]

    # Plotta i dati
    plt.figure(figsize=(10, 6))
    plt.plot(z_sorted, error_sorted, marker='o', linestyle='', alpha=0.7)
    plt.xlabel("Profondità vera (z_true)")
    plt.ylabel("Errore relativo")
    plt.title("Errore relativo rispetto alla profondità")
    plt.grid(True)
    plt.tight_layout()

    # Salva il grafico
    plt.savefig(output_plot)
    print(f"Grafico salvato in: {output_plot}")

    # Mostra il grafico
    plt.show()

# Esempio di utilizzo
true_csv_path = "img/disparity_kitty_one_uvz.csv"  # Percorso al CSV con profondità vere
pred_csv_path = "output/depth_map/kitty_one_pred_depth.csv"  # Percorso al CSV con profondità predette

plot_relative_error(true_csv_path, pred_csv_path)
