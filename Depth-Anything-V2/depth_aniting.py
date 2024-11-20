import os
import cv2
import torch
from depth_anything_v2.dpt import DepthAnythingV2

def generate_depth_map(image_path, output_path, model_checkpoint):
    """
    Genera una mappa di profondità per un'immagine e la salva nel percorso specificato.

    Args:
        image_path (str): Percorso dell'immagine di input.
        output_path (str): Percorso per salvare la mappa di profondità.
        model_checkpoint (str): Percorso del checkpoint del modello.
    """
    # Controlla se il file dell'immagine esiste
    if not os.path.exists(image_path):
        print(f"Errore: Immagine non trovata in {image_path}")
        return

    # Carica il modello Depth-Anything
    print("Caricamento del modello...")
    model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load(model_checkpoint, map_location='cpu'))
    model.eval()
    print("Modello caricato con successo.")

    # Carica l'immagine
    print(f"Caricamento dell'immagine da {image_path}...")
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        print("Errore: Immagine non valida o formato non supportato.")
        return

    # Genera la mappa di profondità
    print("Generazione della mappa di profondità...")
    depth_map = model.infer_image(raw_img)  # HxW raw depth map

    # Normalizza la mappa di profondità per la visualizzazione
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_uint8 = depth_map_normalized.astype("uint8")

    # Salva la mappa di profondità
    print(f"Salvataggio della mappa di profondità in {output_path}...")
    cv2.imwrite(output_path, depth_map_uint8)
    print("Mappa di profondità salvata con successo.")

if __name__ == "__main__":
    # Percorso dell'immagine di input
    image_path = "img/bee.jpg"  # Sostituisci con il percorso della tua immagine

    # Percorso per il checkpoint del modello
    model_checkpoint = "checkpoints/depth_anything_v2_vitl.pth"  # Sostituisci con il percorso corretto

    # Percorso dell'immagine di output
    output_path = os.path.splitext(image_path)[0] + "_depth.jpg"

    # Genera la mappa di profondità
    generate_depth_map(image_path, output_path, model_checkpoint)
