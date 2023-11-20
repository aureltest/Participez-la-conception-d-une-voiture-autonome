from flask import Flask, render_template, request, jsonify
from PIL import Image
import requests
import os
import glob
import base64
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
PREDICTION_API_URL = "https://api-cityscapes.azurewebsites.net/predict"


@app.route('/')
def index():
    """
    Route de l'index. Affiche la page principale de l'application web.
    
    Récupère les chemins des images et des masques, extrait les identifiants des images et renvoie 
    la page 'index.html' avec les identifiants des images pour le rendu.
    """
    ...
    image_paths, _ = get_image_and_mask_paths()
    image_ids = extract_image_ids_from_paths(image_paths)
    return render_template('index.html', image_ids=image_ids)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Route de prédiction. Gère la soumission d'une requête de prédiction d'image.

    Récupère l'identifiant de l'image à partir de la requête, envoie l'image au service de prédiction 
    d'API externe, et traite la réponse pour afficher l'image originale, le masque réel et le masque prédit 
    sur la page 'results.html'.
    """
    image_id = request.form.get('image_id')
    if not image_id:
        return jsonify({"status": "error", "message": "No image ID provided"})

    real_image_path = os.path.join(ROOT_DIR, "images", f"{image_id}_leftImg8bit.png")
    real_mask_path = os.path.join(ROOT_DIR, "mask", f"{image_id}_gtFine_labelIds.png")

    with open(real_image_path, 'rb') as image_file:
        response = requests.post(PREDICTION_API_URL, files={"image": image_file})
        print("Response status code:", response.status_code)
    if response.status_code == 200:
        if response.headers['Content-Type'] == 'image/png':
            predicted_mask_bytes = BytesIO(response.content)

            real_image = Image.open(real_image_path)
            real_image_base64 = image_to_base64(real_image)

            combined_values = get_combined_unique_values(real_mask_path, predicted_mask_bytes)
            colored_real_mask = apply_combined_colormap(real_mask_path, combined_values)
            colored_predicted_mask = apply_combined_colormap(predicted_mask_bytes, combined_values)

            real_mask_base64 = image_to_base64(colored_real_mask)
            predicted_mask_base64 = image_to_base64(colored_predicted_mask)
            return render_template('results.html',
                                   real_image=real_image_base64,
                                   real_mask=real_mask_base64,
                                   predicted_mask=predicted_mask_base64)
        else:
            try:
                prediction_data = response.json()
                return jsonify({"status": "error", "message": prediction_data.get('message', 'Unknown error')})
            except Exception as e:
                return jsonify({"status": "error", "message": f"Failed to parse JSON from prediction API: {str(e)}",
                                "response_content": response.text})
    else:
        return jsonify({"status": "error", "message": f"API returned status code {response.status_code}",
                        "response_content": response.text})


ROOT_DIR = "static/api_sample"


def get_image_and_mask_paths():
    """
    Récupère les chemins des images et des masques dans les répertoires spécifiés.

    Returns:
        Tuple[List[str], Dict[str, str]]: Liste des chemins des images et dictionnaire associant les chemins des images 
        à leurs masques correspondants.
    """
    img_dir = os.path.join(ROOT_DIR, "images")
    label_dir = os.path.join(ROOT_DIR, "mask")

    img_names = sorted([name for name in os.listdir(img_dir) if "_leftImg8bit.png" in name])
    label_names = sorted([name.replace("_leftImg8bit.png", "_gtFine_labelIds.png") for name in img_names])

    image_paths = [os.path.join(img_dir, name) for name in img_names]
    mask_paths = {os.path.join(img_dir, img_name): os.path.join(label_dir, label_name) for img_name, label_name in
                  zip(img_names, label_names)}

    return image_paths, mask_paths


def extract_image_ids_from_paths(image_paths):
    """
    Extrait les identifiants des images à partir de leurs chemins.

    Args:
        image_paths (List[str]): Liste des chemins des images.

    Returns:
        List[str]: Liste des identifiants des images.
    """
    return [os.path.basename(path).split('_leftImg8bit.png')[0] for path in image_paths]


def image_to_base64(img, size=(256, 256)):
    """
    Convertit une image en chaîne de caractères encodée en base64.

    Args:
        img (PIL.Image.Image): Image à convertir.
        size (tuple, optional): Nouvelle taille de l'image. Par défaut à (256, 256).

    Returns:
        str: Image convertie en chaîne base64.
    """
    img = img.resize(size)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_combined_unique_values(*images):
    """
    Récupère les valeurs uniques combinées à partir de plusieurs images.

    Args:
        images (Tuple[PIL.Image.Image, ...]): Images à partir desquelles extraire les valeurs uniques.

    Returns:
        np.ndarray: Tableau de valeurs uniques combinées.
    """
    combined_values = []
    for img_path in images:
        img = Image.open(img_path)
        combined_values.extend(np.unique(np.array(img)))
    return np.unique(combined_values)


def apply_combined_colormap(label_img, combined_values):
    """
    Applique une colormap combinée à une image de masque.

    Args:
        label_img (PIL.Image.Image): Image de masque à colorer.
        combined_values (np.ndarray): Tableau de valeurs uniques pour définir la colormap.

    Returns:
        PIL.Image.Image: Image de masque colorée.
    """
    mask_array = np.array(Image.open(label_img))
    cmap = plt.get_cmap('viridis', len(combined_values))

    value_map = {value: idx for idx, value in enumerate(combined_values)}
    mask_mapped = np.vectorize(value_map.get)(mask_array)

    label_colored = cmap(mask_mapped)
    label_colored = (label_colored[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(label_colored)


if __name__ == "__main__":
    app.run(debug=True)
