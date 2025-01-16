from flask import Flask, render_template, request, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify


# Initialize Flask app
app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model_path = 'model.h5'  # Update with the correct path to your model
model = load_model(model_path)

# Class labels dictionary
class_labels = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry_(including_sour)___Powdery_mildew',
    6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight',
    10: 'Corn_(maize)___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
}
cures = {
    'Apple___Apple_scab': "Remove and destroy fallen leaves and infected fruits during the fall to reduce overwintering spores. Apply fungicides such as captan, mancozeb, or myclobutanil before symptoms appear. Use resistant varieties if available and prune trees properly to improve air circulation.",
    'Apple___Black_rot': "Prune and burn infected branches, twigs, and mummified fruits to prevent the spread of spores. Use fungicides containing thiophanate-methyl or captan during the growing season. Maintain proper tree health with adequate fertilization and watering.",
    'Apple___Cedar_apple_rust': "Remove nearby juniper or cedar trees, as they act as alternate hosts for the fungus. Apply fungicides like myclobutanil or mancozeb in early spring during the bud and flowering stages. Opt for resistant apple varieties and ensure proper pruning for airflow.",
    'Apple___healthy': "Maintain good cultural practices, including regular pruning, watering, and fertilization. Monitor for any signs of pests or diseases to ensure continued health.",
    'Blueberry___healthy': "Ensure proper watering, avoid waterlogging, and maintain an acidic soil pH of 4.5-5.5. Provide mulch to retain soil moisture and monitor for potential pests or diseases.",
    'Cherry_(including_sour)___Powdery_mildew': "Prune affected branches and ensure good air circulation around trees. Apply fungicides like sulfur or myclobutanil at the first sign of symptoms. Water the base of the plant to avoid wetting foliage.",
    'Cherry_(including_sour)___healthy': "Maintain proper care with regular watering, fertilization, and pruning. Protect against pests and monitor for signs of diseases.",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Rotate crops with non-host plants to reduce inoculum in the soil. Use resistant hybrids and apply fungicides like azoxystrobin or pyraclostrobin when conditions favor disease development.",
    'Corn_(maize)__Common_rust': "Plant resistant hybrids and monitor fields regularly for symptoms. Apply fungicides like mancozeb or chlorothalonil if infection is severe. Maintain proper crop spacing for air circulation.",
    'Corn_(maize)___Northern_Leaf_Blight': "Use resistant hybrids and rotate crops with non-host species. Apply fungicides such as azoxystrobin or propiconazole if symptoms appear early in the growing season.",
    'Corn_(maize)___healthy': "Ensure proper irrigation, fertilization, and pest management. Regularly monitor for early signs of disease.",
    'Grape___Black_rot': "Remove infected leaves, fruits, and stems from the vineyard to prevent spore spread. Apply fungicides like mancozeb, captan, or myclobutanil during the growing season. Use resistant grape varieties if available.",
    'Grape__Esca(Black_Measles)': "Prune infected vines and disinfect pruning tools to prevent spread. Avoid overwatering and apply fungicides like thiophanate-methyl or benomyl as preventive measures.",
    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)': "Ensure proper vineyard sanitation by removing infected leaves. Apply fungicides containing copper or mancozeb during the early stages of infection. Improve air circulation by pruning.",
    'Grape___healthy': "Provide balanced fertilization, proper watering, and regular monitoring for pests and diseases. Ensure good vineyard sanitation practices.",
    'Orange__Haunglongbing(Citrus_greening)': "Remove and destroy infected trees to prevent the spread of the bacteria. Control the Asian citrus psyllid vector using insecticides. Use disease-free planting material and implement nutrient management to support tree health.",
    'Peach___Bacterial_spot': "Plant resistant varieties and apply copper-based bactericides during early leaf development. Prune and destroy infected branches to reduce bacterial load. Avoid overhead irrigation.",
    'Peach___healthy': "Maintain a regular schedule of watering, fertilization, and pruning. Monitor for signs of pests or diseases.",
    'Pepper,bell__Bacterial_spot': "Use resistant varieties and apply copper-based bactericides. Avoid overhead irrigation and ensure proper crop rotation. Remove and destroy infected plants.",
    'Pepper,bell__healthy': "Maintain optimal growing conditions with adequate sunlight, watering, and fertilization. Regularly monitor for pests and diseases.",
    'Potato___Early_blight': "Practice crop rotation and remove infected plant debris from fields. Apply fungicides like chlorothalonil or mancozeb at the first sign of symptoms. Use certified disease-free seeds.",
    'Potato___Late_blight': "Destroy infected plants and tubers immediately. Use fungicides like chlorothalonil or copper-based products. Practice crop rotation and avoid planting potatoes near tomatoes.",
    'Potato___healthy': "Ensure proper soil health and provide regular watering and fertilization. Monitor for pests and diseases.",
    'Raspberry___healthy': "Provide good air circulation through proper spacing and pruning. Ensure regular monitoring and apply pest and disease control measures as needed.",
    'Soybean___healthy': "Rotate crops, ensure proper soil health, and monitor for potential diseases. Apply preventive fungicides if needed.",
    'Squash___Powdery_mildew': "Remove infected leaves and avoid overhead watering. Apply fungicides like sulfur or neem oil to control the disease. Ensure good air circulation by spacing plants adequately.",
    'Strawberry___Leaf_scorch': "Remove infected leaves and apply fungicides like captan or mancozeb. Maintain good irrigation practices and avoid overhead watering.",
    'Strawberry___healthy': "Provide adequate irrigation and fertilization. Regularly inspect plants for pests and diseases.",
    'Tomato___Bacterial_spot': "Use resistant varieties and apply copper-based sprays. Avoid overhead irrigation and remove infected plants to reduce bacterial spread.",
    'Tomato___Early_blight': "Remove infected plant debris and apply fungicides like chlorothalonil or mancozeb. Practice crop rotation and avoid overcrowding plants.",
    'Tomato___Late_blight': "Destroy infected plants and apply fungicides like chlorothalonil or copper-based products. Ensure good air circulation and avoid overhead irrigation.",
    'Tomato___Leaf_Mold': "Prune affected leaves and apply fungicides like chlorothalonil or copper-based sprays. Maintain good ventilation in greenhouses.",
    'Tomato___Septoria_leaf_spot': "Remove and destroy infected leaves. Apply fungicides containing chlorothalonil or mancozeb. Avoid overhead watering and ensure crop rotation.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Use miticides or insecticidal soaps to control mites. Maintain adequate humidity and water plants properly to reduce mite infestations.",
    'Tomato___Target_Spot': "Apply fungicides like azoxystrobin or chlorothalonil at the first sign of symptoms. Remove infected leaves and improve plant spacing for airflow.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Control whitefly populations with insecticides or reflective mulches. Use resistant varieties and remove infected plants promptly.",
    'Tomato___Tomato_mosaic_virus': "Remove and destroy infected plants. Disinfect tools and equipment with a bleach solution. Use disease-free seeds and resistant varieties.",
    'Tomato___healthy': "Ensure proper watering, fertilization, and pest management. Regularly monitor plants for early signs of diseases."
}
# Helper function to check allowed files
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def home():
    return render_template('index.html', image_url=None, result=None, cure=None, error=None)

# Updated /predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify(error="No file part in the request."), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No file selected."), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the image and make predictions
        image_array = preprocess_image(file_path)
        prediction = model.predict(image_array, verbose=0)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_labels.get(predicted_class_index, "Unknown")
        confidence = float(np.max(prediction) * 100)  # Convert confidence to percentage

        # Get the cure for the predicted disease
        cure = cures.get(predicted_class_name, "No cure information available.")

        return jsonify(result=predicted_class_name, confidence=f"{confidence:.2f}%", cure=cure)
    else:
        return jsonify(error="Only image files are supported."), 400


if __name__ == '__main__':
    app.run(debug=True)
