# app.py - Aplicación Flask para el clasificador VGG16

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import json
import os
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

# Configuración
MODEL_PATH = 'models/vgg16_final_model.h5'
RESULTS_PATH = 'results/model_results.json'
IMG_SIZE = (224, 224)

# Cargar modelo y configuración
print("Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✓ Modelo cargado")

# Cargar resultados y configuración
with open(RESULTS_PATH, 'r') as f:
    model_info = json.load(f)

CLASSES = model_info['classes']
LABEL_MAP = model_info['label_map']

# Estadísticas de normalización (deben coincidir con el entrenamiento)
with open('processed_dataset/normalization_stats.json', 'r') as f:
    norm_stats = json.load(f)
    MEAN_RGB = np.array(norm_stats['mean_rgb']) / 255.0
    STD_RGB = np.array(norm_stats['std_rgb']) / 255.0

def preprocess_image(image_bytes):
    """
    Preprocesa una imagen para predicción
    """
    # Convertir bytes a imagen
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convertir a RGB si es necesario
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convertir a numpy array
    img_array = np.array(image)
    
    # Redimensionar
    img_resized = cv2.resize(img_array, IMG_SIZE)
    
    # Escalar a [0, 1]
    img_float = img_resized.astype(np.float32) / 255.0
    
    # Normalizar
    img_normalized = (img_float - MEAN_RGB) / STD_RGB
    
    # Añadir dimensión de batch
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, img_array

def predict_image(image_bytes):
    """
    Realiza predicción en una imagen
    """
    # Preprocesar
    img_preprocessed, img_original = preprocess_image(image_bytes)
    
    # Predicción
    predictions = model.predict(img_preprocessed, verbose=0)
    
    # Obtener clase y confianza
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    predicted_class = CLASSES[predicted_class_idx]
    
    # Probabilidades para todas las clases
    probabilities = {
        CLASSES[i]: float(predictions[0][i]) 
        for i in range(len(CLASSES))
    }
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities,
        'all_predictions': predictions[0].tolist()
    }

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html', 
                         classes=CLASSES,
                         model_info=model_info)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint para predicción"""
    try:
        # Verificar que se envió una imagen
        if 'image' not in request.files:
            return jsonify({'error': 'No se envió ninguna imagen'}), 400
        
        file = request.files['image']
        
        # Verificar que el archivo no está vacío
        if file.filename == '':
            return jsonify({'error': 'Archivo vacío'}), 400
        
        # Leer bytes de la imagen
        image_bytes = file.read()
        
        # Realizar predicción
        result = predict_image(image_bytes)
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model-info')
def api_model_info():
    """Retorna información del modelo"""
    return jsonify(model_info)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': CLASSES
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)