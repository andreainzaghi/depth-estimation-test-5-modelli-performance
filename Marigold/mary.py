import logging
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from marigold import MarigoldPipeline

# -------------------- Configurazioni fisse --------------------
CHECKPOINT_PATH = "checkpoint/marigold-v1-0"  # Percorso predefinito del checkpoint
INPUT_RGB_DIR = "img"  # Cartella predefinita delle immagini
OUTPUT_DIR = "output/depth_map"  # Cartella predefinita per l'output

DENOISE_STEPS = 4  # Passi di denoising
ENSEMBLE_SIZE = 5  # Ensemble size
HALF_PRECISION = False  # Precisione metà (fp16)

PROCESSING_RES = None  # Risoluzione massima di elaborazione
MATCH_INPUT_RES = True  # Match con la risoluzione originale
RESAMPLE_METHOD = "bilinear"  # Metodo di ridimensionamento
COLOR_MAP = "Spectral"  # Mappa colori per la depth map
SEED = None  # Nessun seed per riproducibilità
BATCH_SIZE = 0  # Batch size automatico
APPLE_SILICON = False  # Supporto per Apple Silicon

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]  # Tipi di file accettati

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO)

# -------------------- Device --------------------
if APPLE_SILICON:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps:0")
    else:
        device = torch.device("cpu")
        logging.warning("MPS is not available. Running on CPU will be slow.")
else:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
logging.info(f"Device = {device}")

# -------------------- Preparazione --------------------
# Creazione delle cartelle di output
output_dir_color = os.path.join(OUTPUT_DIR, "depth_colored")
output_dir_tif = os.path.join(OUTPUT_DIR, "depth_bw")
output_dir_npy = os.path.join(OUTPUT_DIR, "depth_npy")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(output_dir_color, exist_ok=True)
os.makedirs(output_dir_tif, exist_ok=True)
os.makedirs(output_dir_npy, exist_ok=True)
logging.info(f"Output dir = {OUTPUT_DIR}")

# -------------------- Caricamento immagini --------------------
rgb_filename_list = glob(os.path.join(INPUT_RGB_DIR, "*"))
rgb_filename_list = [
    f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
]
rgb_filename_list = sorted(rgb_filename_list)
n_images = len(rgb_filename_list)
if n_images > 0:
    logging.info(f"Found {n_images} images")
else:
    logging.error(f"No image found in '{INPUT_RGB_DIR}'")
    exit(1)

# -------------------- Caricamento modello --------------------
dtype = torch.float16 if HALF_PRECISION else torch.float32
variant = "fp16" if HALF_PRECISION else None

pipe: MarigoldPipeline = MarigoldPipeline.from_pretrained(
    CHECKPOINT_PATH, variant=variant, torch_dtype=dtype
)

try:
    pipe.enable_xformers_memory_efficient_attention()
except ImportError:
    pass  # Procedi senza xformers

pipe = pipe.to(device)
logging.info(
    f"scale_invariant: {pipe.scale_invariant}, shift_invariant: {pipe.shift_invariant}"
)

logging.info(
    f"Inference settings: checkpoint = `{CHECKPOINT_PATH}`, "
    f"denoise_steps = {DENOISE_STEPS}, "
    f"ensemble_size = {ENSEMBLE_SIZE}, "
    f"processing resolution = {PROCESSING_RES or pipe.default_processing_resolution}, "
    f"seed = {SEED}; "
    f"color_map = {COLOR_MAP}."
)

# -------------------- Inference --------------------
with torch.no_grad():
    for rgb_path in tqdm(rgb_filename_list, desc="Estimating depth", leave=True):
        input_image = Image.open(rgb_path)

        # Generatore casuale
        generator = None
        if SEED is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(SEED)

        # Predizione
        pipe_out = pipe(
            input_image,
            denoising_steps=DENOISE_STEPS,
            ensemble_size=ENSEMBLE_SIZE,
            processing_res=PROCESSING_RES,
            match_input_res=MATCH_INPUT_RES,
            batch_size=BATCH_SIZE,
            color_map=COLOR_MAP,
            show_progress_bar=True,
            resample_method=RESAMPLE_METHOD,
            generator=generator,
        )

        depth_pred: np.ndarray = pipe_out.depth_np
        depth_colored: Image.Image = pipe_out.depth_colored

        # Salvataggio
        rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
        pred_name_base = rgb_name_base + "_pred"

        # Salva npy
        npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
        np.save(npy_save_path, depth_pred)

        # Salva 16-bit uint png
        depth_to_save = (depth_pred * 65535.0).astype(np.uint16)
        png_save_path = os.path.join(output_dir_tif, f"{pred_name_base}.png")
        Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")

        # Salva immagine colorata
        colored_save_path = os.path.join(output_dir_color, f"{pred_name_base}_colored.png")
        depth_colored.save(colored_save_path)

logging.info("Depth estimation completed successfully!")
