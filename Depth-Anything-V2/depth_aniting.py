import os
import cv2
import torch
from depth_anything_v2.dpt import DepthAnythingV2
import numpy as np
import pandas as pd

def generate_depth_maps(input_folder, output_folder, model_checkpoint):
    """
    Genera mappe di profondità per immagini specifiche in sottocartelle e le salva nella cartella di output.

    Args:
        input_folder (str): Percorso della cartella principale contenente le sottocartelle con immagini.
        output_folder (str): Percorso della cartella per salvare le mappe di profondità.
        model_checkpoint (str): Percorso del checkpoint del modello.
    """
    # Controlla se la cartella di input esiste
    if not os.path.exists(input_folder):
        print(f"Errore: Cartella di input non trovata in {input_folder}")
        return
    
    # Crea la cartella di output se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Carica il modello Depth-Anything
    print("Caricamento del modello...")
    model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load(model_checkpoint, map_location='cpu'))
    model.eval()
    print("Modello caricato con successo.")

    # Itera attraverso tutte le sottocartelle
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        
        # Controlla se è una directory
        if not os.path.isdir(subfolder_path):
            continue

        # Trova l'immagine desiderata nella sottocartella
        target_image = None
        for file_name in sorted(os.listdir(subfolder_path)):
            if "groundtruth" not in file_name and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                target_image = file_name
                break  # Prendi solo la prima immagine valida

        if target_image is None:
            print(f"Nessuna immagine valida trovata nella cartella {subfolder}")
            continue

        input_path = os.path.join(subfolder_path, target_image)

        # Carica l'immagine
        print(f"Caricamento dell'immagine da {input_path}...")
        raw_img = cv2.imread(input_path)
        if raw_img is None:
            print(f"Errore: Immagine {target_image} non valida o formato non supportato.")
            continue

        # Genera la mappa di profondità
        print(f"Generazione della mappa di profondità per {target_image}...")
        depth_map = model.infer_image(raw_img)  # HxW raw depth map

        # Normalizza la mappa di profondità per la visualizzazione
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_uint8 = depth_map_normalized.astype("uint8")

        # Percorso di output per la mappa di profondità
        output_image_path = os.path.join(output_folder, f"{subfolder}_{os.path.splitext(target_image)[0]}_depth.jpg")
        print(f"Salvataggio della mappa di profondità in {output_image_path}...")
        cv2.imwrite(output_image_path, depth_map_uint8)

        # Converte la mappa di profondità in un array numpy e salva come CSV
        print(f"Conversione della mappa di profondità di {target_image} in un array numpy e salvataggio come CSV...")
        h, w = depth_map.shape
        uvz_data = []
        for u in range(h):
            for v in range(w):
                uvz_data.append([u, v, depth_map[u, v]])

        # Salvataggio in formato CSV
        csv_output_path = os.path.join(output_folder, f"{subfolder}_{os.path.splitext(target_image)[0]}_depth.csv")
        uvz_array = np.array(uvz_data)
        pd.DataFrame(uvz_array, columns=['u', 'v', 'z']).to_csv(csv_output_path, index=False)
        print(f"File CSV salvato in {csv_output_path}.")

if __name__ == "__main__":
    # Percorso della cartella principale di input
    input_folder = "img&val"  # Sostituisci con il percorso della tua cartella

    # Percorso della cartella di output
    output_folder = "output_depth_maps"  # Sostituisci con il percorso desiderato per i file di output

    # Percorso per il checkpoint del modello
    model_checkpoint = "checkpoints/depth_anything_v2_vitl.pth"  # Sostituisci con il percorso corretto

    # Genera le mappe di profondità
    generate_depth_maps(input_folder, output_folder, model_checkpoint)
