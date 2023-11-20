from flask import Flask, request, jsonify, send_file
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import segmentation_models as sm
from io import BytesIO
import zipfile
from azure.storage.blob import BlobServiceClient
import base64

jaccard_index = sm.metrics.IOUScore()
dice_coef = sm.metrics.FScore(beta=1)
class_weights = [0.04681816, 0.01256017, 0.02227352, 0.27617385, 0.03223511,
                 0.13659585, 0.40692867, 0.06641466]

dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
custom_objects = {"iou_score": jaccard_index,
                  "f1-score": dice_coef,
                  "dice_loss": dice_loss}

app = Flask(__name__)

def download_model():
    """
    Télécharge un modèle pré-entraîné pour la segmentation d'images depuis Azure Blob Storage.
    
    Cette fonction crée une connexion à Azure Blob Storage, télécharge le modèle zippé, et l'extrait 
    dans un dossier spécifié.
    
    Exceptions:
        Lève une exception si le téléchargement ou l'extraction échouent.
    """
    try:
        # Créer le client BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(
            "DefaultEndpointsProtocol=https;AccountName=segmodels;AccountKey=3tiu6DQpY7UMvRkQBSkMY1E8rnQTZCsiLCQrDw9pqnxHFEQLOATwJrkStmmMw+ilHz5jLh04Cy2B+ASt7wiWSw==;EndpointSuffix=core.windows.net")

        # Spécifiez le nom de votre conteneur et le nom du blob pour votre modèle tflite
        container_name = 'fpn-conteneur'
        blob_name = 'FPN_efficientnetb3_aug.zip'

        # Créer le BlobClient
        blob_client = blob_service_client.get_blob_client(container_name, blob_name)

        # Télécharger le blob en tant que fichier
        with open("FPN_efficientnetb3_aug.zip", "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
            with zipfile.ZipFile("FPN_efficientnetb3_aug.zip", 'r') as zip_ref:
                zip_ref.extractall("model")
    except Exception as ex:
        print('Exception:')
        print(ex)


model_path = "model/FPN_efficientnetb3_aug"
if not os.path.isdir(model_path):
    download_model()

model = load_model("model/FPN_efficientnetb3_aug", custom_objects=custom_objects)


def resize(image):
    """
    Redimensionne l'image fournie à une taille spécifique.

    Args:
        image (bytes): Image sous forme de bytes.

    Returns:
        PIL.Image.Image: Image redimensionnée.
    """
    image = Image.open(BytesIO(image))
    image = image.resize((256, 256))
    return image


def processing(resized_image):
    """
    Traite l'image redimensionnée pour la préparation de la prédiction.

    Args:
        resized_image (PIL.Image.Image): Image redimensionnée.

    Returns:
        np.ndarray: Image sous forme de tableau NumPy, normalisée et prête pour la prédiction.
    """
    image_np = np.array(resized_image) / 255.0
    image_np = np.expand_dims(image_np, axis=0)
    return image_np


def convert_prediction(mask):
    """
    Convertit le masque de prédiction en un masque de classe unique.

    Args:
        mask (np.ndarray): Masque prédit par le modèle.

    Returns:
        np.ndarray: Masque de classe unique.
    """
    single_class_mask = np.argmax(mask, axis=-1)
    return single_class_mask


@app.route("/predict", methods=["POST"])
def predict():
    """
    Route pour la prédiction de segmentation d'images.

    Cette route reçoit une image via une requête POST, la traite, effectue une prédiction de segmentation 
    avec le modèle chargé, et renvoie le masque de segmentation prédit.

    Returns:
        Flask.Response: Image du masque de segmentation prédite en cas de succès, 
                        ou un message d'erreur en cas d'échec.
    """
    try:
        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "No image provided"}), 400
        image = request.files['image'].read()

        if len(image) > 10 * 1024 * 1024:
            return jsonify({"status": "error", "message": "Image too large. Maximum size allowed is 10MB"}), 400

        image_format = Image.open(BytesIO(image)).format
        if image_format not in ['JPEG', 'PNG']:
            return jsonify({"status": "error",
                            "message": f"Image format '{image_format}' not supported. Please use JPEG or PNG"}), 400

        image = resize(image)
        image = processing(image)
        prediction = model.predict(image)
        prediction = convert_prediction(prediction)
        prediction = np.squeeze(prediction, axis=0)
        predicted_mask_img = Image.fromarray(prediction.astype(np.uint8))

        byte_io = BytesIO()
        predicted_mask_img.save(byte_io, 'PNG')
        byte_io.seek(0)

        return send_file(byte_io, mimetype='image/png'), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
