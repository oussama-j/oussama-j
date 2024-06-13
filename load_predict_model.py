import tensorflow as tf
import numpy as np
from PIL import Image

# Fonction de prédiction pour une nouvelle image
def predict_image(image_path, model):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print(f'Chargement de l\'image {image_path}...')
    # Chargement et pré-traitement de l'image
    img = Image.open(image_path)
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension pour le batch

    print("Prédiction de l'image...")
    # Prédiction
    with tf.device('/GPU:0'):
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
    
    print(f'Classe prédite: {class_names[predicted_class]}')

# Charger le modèle sauvegardé et prédire une nouvelle image
def load_model_and_predict(image_path):
    # Chargement du modèle sauvegardé
    print("Chargement du modèle sauvegardé...")
    model = tf.keras.models.load_model('cifar10_cnn_model.h5')
    
    # Prédire une nouvelle image
    predict_image(image_path, model)

# Exemple d'utilisation pour prédire une image après chargement du modèle sauvegardé
load_model_and_predict('path_to_your_image.jpg')
