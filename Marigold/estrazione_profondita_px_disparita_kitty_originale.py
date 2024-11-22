import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_uvz_from_disparity(disparity_img_path, focal_length, baseline, disparity_scale, output_csv_path, show_depth_preview=False):
    """
    Genera un file CSV contenente (u, v, z) da una mappa di disparità KITTI.

    Args:
        disparity_img_path (str): Percorso all'immagine di disparità.
        focal_length (float): Lunghezza focale in pixel.
        baseline (float): Distanza tra le telecamere in metri.
        disparity_scale (float): Fattore di scala della disparità.
        output_csv_path (str): Percorso al file CSV di output.
        show_depth_preview (bool): Se True, mostra l'anteprima della mappa di profondità.
    """
    # Carica l'immagine di disparità
    disparity = cv2.imread(disparity_img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if disparity is None:
        raise ValueError(f"Errore nel caricamento dell'immagine di disparità: {disparity_img_path}")

    # Stampa i valori minimi e massimi della disparità
    print(f"Elaborando {disparity_img_path} | Valori disparità: Min={np.min(disparity)}, Max={np.max(disparity)}")

    # Scala la disparità per riportarla ai valori effettivi
    disparity_scaled = disparity / disparity_scale
    # Sostituisci disparità nulle o negative con un valore minimo per evitare divisioni per zero
    disparity_scaled[disparity_scaled <= 0] = 1e-5

    # Calcolo della profondità (z) in metri
    z = (baseline * focal_length) / disparity_scaled

    # Genera le coordinate (u, v) dei pixel
    height, width = disparity.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Maschera per selezionare i pixel con valori di disparità validi
    valid_mask = disparity_scaled > 1e-5

    # Combina (u, v, z) per i pixel validi in un unico array
    points = np.column_stack((u[valid_mask], v[valid_mask], z[valid_mask]))

    # Salva i punti validi in un file CSV
    df = pd.DataFrame(points, columns=['u', 'v', 'z'])
    df.to_csv(output_csv_path, index=False)
    print(f"File CSV generato: {output_csv_path}")

    # Mostra un'anteprima della mappa di profondità se richiesto
    if show_depth_preview:
        plt.imshow(z, cmap='viridis')
        plt.colorbar(label="Profondità (m)")
        plt.title(f"Mappa di profondità: {os.path.basename(disparity_img_path)}")
        plt.show()


def process_disparity_folder(input_folder, output_folder, focal_length, baseline, disparity_scale, show_depth_preview=False):
    """
    Processa tutte le immagini di disparità in una cartella e salva i CSV di output in un'altra cartella.

    Args:
        input_folder (str): Cartella contenente le immagini di disparità.
        output_folder (str): Cartella in cui salvare i file CSV di output.
        focal_length (float): Lunghezza focale in pixel.
        baseline (float): Distanza tra le telecamere in metri.
        disparity_scale (float): Fattore di scala per la mappa di disparità.
        show_depth_preview (bool): Se True, mostra l'anteprima della mappa di profondità.
    """
    # Crea la cartella di output se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Trova tutte le immagini di disparità nella cartella di input
    supported_formats = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    disparity_images = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_formats)]

    if not disparity_images:
        print(f"Nessuna immagine trovata nella cartella: {input_folder}")
        return

    # Processa ogni immagine
    for disparity_image in disparity_images:
        input_path = os.path.join(input_folder, disparity_image)
        output_csv_path = os.path.join(output_folder, os.path.splitext(disparity_image)[0] + "_uvz.csv")
        generate_uvz_from_disparity(
            input_path,
            focal_length,
            baseline,
            disparity_scale,
            output_csv_path,
            show_depth_preview=show_depth_preview
        )


if __name__ == "__main__":
    # Parametri di calibrazione del dataset KITTI
    focal_length = 721.5377    # Lunghezza focale in pixel
    baseline = 0.54            # Baseline tra le telecamere in metri
    disparity_scale = 256.0    # Fattore di scala per la mappa di disparità

    # Percorsi delle cartelle
    input_folder = "dispari"  # Cartella contenente le immagini di disparità
    output_folder = "csv_uvz_output"  # Cartella in cui salvare i file CSV

    # Esegui la funzione
    process_disparity_folder(
        input_folder,
        output_folder,
        focal_length,
        baseline,
        disparity_scale,
        show_depth_preview=False  # Cambia a True per visualizzare le mappe di profondità
    )
