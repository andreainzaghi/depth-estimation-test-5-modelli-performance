import os
from PIL import Image
import torch
import depth_pro
from depth_pro import DepthProConfig, create_model_and_transforms

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

    print("Caricamento dell'immagine...")
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    print("Inferenza in corso...")
    prediction = model.infer(image)
    depth = prediction["depth"].cpu().numpy()

    # Normalizzazione e salvataggio della mappa di profondità
    depth_image = Image.fromarray((depth * 255 / depth.max()).astype("uint8"))
    depth_image.save(output_path)
    print(f"Mappa di profondità salvata in: {output_path}")

if __name__ == "__main__":
    # Percorsi configurabili
    image_path = "img/bee.jpg"  # Percorso dell'immagine di input
    output_path = "img/bee_depth.png"  # Percorso per salvare la mappa di profondità

    create_depth_map(image_path, output_path)
