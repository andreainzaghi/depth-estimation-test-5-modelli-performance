import os
from PIL import Image
import torch
import depth_pro
from depth_pro import DepthProConfig, create_model_and_transforms
import pandas as pd


def create_depth_map(image_path, output_path):
    """
    Genera una mappa di profondità per un'immagine e la salva.

    Args:
        image_path (str): Percorso dell'immagine di input.
        output_path (str): Percorso per salvare la mappa di profondità.
    """
    # Configurazione del modello
    config = DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri="./checkpoints/depth_pro.pt",  # Percorso corretto al checkpoint
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_384",
    )

    print("Caricamento del modello...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = create_model_and_transforms(config=config, device=device)
    model.eval()

    # Caricamento e trasformazione dell'immagine
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    # Inferenza
    prediction = model.infer(image)
    depth = prediction["depth"].cpu().numpy()

    # Normalizzazione e salvataggio della mappa di profondità
    depth_image = Image.fromarray((depth * 255 / depth.max()).astype("uint8"))
    depth_image.save(output_path)

    # Generazione del file CSV
    csv_output_path = os.path.splitext(output_path)[0] + "_depth.csv"
    if depth.ndim == 3:
        depth = depth[0]  # Rimuove la dimensione batch se presente
    height, width = depth.shape
    uvz_data = [[u, v, depth[u, v]] for u in range(height) for v in range(width)]

    # Salvataggio del file CSV
    pd.DataFrame(uvz_data, columns=["u", "v", "z"]).to_csv(csv_output_path, index=False)
    print(f"Mappa di profondità salvata in: {output_path}")
    print(f"File CSV salvato in: {csv_output_path}")


def process_folder(input_folder, output_folder):
    """
    Processa tutte le immagini in una cartella e genera le mappe di profondità corrispondenti.

    Args:
        input_folder (str): Percorso della cartella di input.
        output_folder (str): Percorso della cartella di output.
    """
    # Crea la cartella di output se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Trova tutte le immagini nella cartella di input
    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_formats)]

    if not images:
        print(f"Nessuna immagine trovata nella cartella: {input_folder}")
        return

    # Caricamento e inferenza per ogni immagine
    for image_name in images:
        image_path = os.path.join(input_folder, image_name)
        output_path = os.path.join(output_folder, os.path.splitext(image_name)[0] + "_depth.png")
        print(f"Elaborazione: {image_path}")
        create_depth_map(image_path, output_path)


if __name__ == "__main__":
    # Percorsi configurabili
    input_folder = "img&val"  # Cartella contenente le immagini di input
    output_folder = "output_depth_maps"  # Cartella per salvare le mappe di profondità

    process_folder(input_folder, output_folder)
