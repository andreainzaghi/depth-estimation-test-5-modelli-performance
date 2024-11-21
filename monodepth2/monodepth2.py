# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import glob
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR


def test_simple():
    """Function to predict for a single image or folder of images with default values"""
    # Definizione dei valori di default
    image_path = "assets/test_image.jpg"  # Percorso predefinito all'immagine
    model_name = "mono_640x192"  # Nome predefinito del modello
    ext = "jpg"  # Estensione predefinita delle immagini
    no_cuda = False  # Usa CUDA se disponibile
    pred_metric_depth = False  # Disabilita predizione della profondità metrica

    # Configura il dispositivo (CPU o GPU)
    if torch.cuda.is_available() and not no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if pred_metric_depth and "stereo" not in model_name:
        print("Warning: The pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not be in metric space.")

    # Scarica il modello predefinito se non è già disponibile
    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("models", model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # Caricamento del modello pre-addestrato
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # Estrai altezza e larghezza dell'immagine utilizzate per addestrare il modello
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # Trova immagini di input
    if os.path.isfile(image_path):
        # Test su una singola immagine
        paths = [image_path]
        output_directory = os.path.dirname(image_path)
    elif os.path.isdir(image_path):
        # Cerca immagini nella cartella
        paths = glob.glob(os.path.join(image_path, '*.{}'.format(ext)))
        output_directory = image_path
    else:
        raise Exception("Cannot find image_path: {}".format(image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # Predizione per ogni immagine
    with torch.no_grad():
        for idx, img_path in enumerate(paths):

            if img_path.endswith("_disp.jpg"):
                # Non prevedere per immagini di disparità
                continue

            # Carica e pre-elabora l'immagine
            input_image = pil.open(img_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # Predizione
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Salvataggio del file numpy
            output_name = os.path.splitext(os.path.basename(img_path))[0]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            if pred_metric_depth:
                name_dest_npy = os.path.join(output_directory, "{}_depth.npy".format(output_name))
                metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
                np.save(name_dest_npy, metric_depth)
            else:
                name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
                np.save(name_dest_npy, scaled_disp.cpu().numpy())

                            # Generazione del file CSV
            print("   Generazione del file CSV...")
            csv_output_path = os.path.join(output_directory, "{}_depth.csv".format(output_name))
            depth_np = depth.squeeze().cpu().numpy() if pred_metric_depth else scaled_disp.squeeze().cpu().numpy()
            h, w = depth_np.shape
            uvz_data = []

            for u in range(h):
                for v in range(w):
                    uvz_data.append([u, v, depth_np[u, v]])

            # Salva il CSV
            import pandas as pd
            pd.DataFrame(uvz_data, columns=['u', 'v', 'z']).to_csv(csv_output_path, index=False)
            print("   - {}".format(csv_output_path))


            # Salvataggio immagine colormap
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
            print("   - {}".format(name_dest_npy))

    print('-> Done!')


if __name__ == '__main__':
    test_simple()
