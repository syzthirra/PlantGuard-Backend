import os
import io
import cv2
import base64
import logging
import matplotlib.cm as cm
import imghdr
import gc
import numpy as np

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image

from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    InputLayer,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array

#########################################################
# Flask Configuration
#########################################################

app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

#########################################################
# Model Configuration
#########################################################


IMAGE_SIZE = (224,224)

APP_NAME = "PlantGuard"

VERSION = "1.0.0"

AUTHOR = "Syaza Athirah Sanusi"

#########################################################
# Build CNN Model
#########################################################

WEIGHTS_PATH = "plantguard_weights.weights.h5"


def build_model():

    inputs = Input(shape=(224, 224, 3))

    x = Conv2D(32, (3,3), activation="relu")(inputs)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(64, (3,3), activation="relu")(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(128, (3,3), activation="relu", name="last_conv")(x)
    x = MaxPooling2D((2,2))(x)

    x = Flatten()(x)

    x = Dense(128, activation="relu")(x)

    x = Dropout(0.5)(x)

    outputs = Dense(10, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


try:

    model = build_model()

    model.load_weights(WEIGHTS_PATH)

    logging.info("CNN weights loaded successfully.")

except Exception as e:

    logging.exception("Unable to load CNN weights.")

    raise


#########################################################
# Last Convolution Layer
#########################################################

LAST_CONV_LAYER = None

for layer in reversed(model.layers):

    if isinstance(layer, tf.keras.layers.Conv2D):

        LAST_CONV_LAYER = "last_conv"

        break

logging.info(f"Last Conv Layer : {LAST_CONV_LAYER}")

#########################################################
# Build Model Before GradCAM
#########################################################

def make_gradcam_heatmap(image):

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(LAST_CONV_LAYER).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(image)

        class_index = tf.argmax(predictions[0])

        loss = predictions[:, class_index]

    gradients = tape.gradient(loss, conv_outputs)

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)

    if max_value > 0:
        heatmap /= max_value

    heatmap = heatmap.numpy()

    heatmap = cv2.GaussianBlur(
        heatmap,
        (7, 7),
        0
    )

    del grad_model
    gc.collect()

    return heatmap


def overlay_heatmap(original_image, heatmap):

    original = np.array(original_image.convert("RGB"))

    heatmap = cv2.resize(
        heatmap,
        (original.shape[1], original.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        original,
        0.65,
        heatmap,
        0.35,
        0
    )

    return overlay

def image_to_base64(image):

    image = Image.fromarray(image)

    buffer = io.BytesIO()

    image.save(
       buffer,
       format="PNG",
       optimize=True
)

    return base64.b64encode(

        buffer.getvalue()

    ).decode("utf-8")



CLASS_NAMES = [

    "Tomato_Bacterial_spot",

    "Tomato_Early_blight",

    "Tomato_Late_blight",

    "Tomato_Leaf_Mold",

    "Tomato_Septoria_leaf_spot",

    "Tomato_Spider_mites_Two_spotted_spider_mite",

    "Tomato_Target_Spot",

    "Tomato_Tomato_YellowLeaf_Curl_Virus",

    "Tomato_Tomato_mosaic_virus",

    "Tomato_healthy"

]

RECOMMENDATIONS = {

    "Tomato_Bacterial_spot":
    "Remove infected leaves. Avoid overhead watering. Apply copper-based bactericide.",

    "Tomato_Early_blight":
    "Apply fungicide. Remove infected foliage. Rotate crops.",

    "Tomato_Late_blight":
    "Destroy infected plants immediately. Apply protective fungicide.",

    "Tomato_Leaf_Mold":
    "Improve ventilation. Reduce humidity inside greenhouse.",

    "Tomato_Septoria_leaf_spot":
    "Remove infected leaves. Apply fungicide regularly.",

    "Tomato_Spider_mites_Two_spotted_spider_mite":
    "Use miticide or insecticidal soap. Maintain proper humidity.",

    "Tomato_Target_Spot":
    "Apply fungicide. Improve air circulation.",

    "Tomato_Tomato_YellowLeaf_Curl_Virus":
    "Control whiteflies immediately. Remove infected plants.",

    "Tomato_Tomato_mosaic_virus":
    "Destroy infected plants and disinfect gardening tools.",

    "Tomato_healthy":
    "Plant appears healthy. Continue regular watering and monitoring."

}

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image = img_to_array(image)

    image = image.astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)

    return image




def predict_disease(image):

    processed = preprocess_image(image)

    prediction = model.predict(processed,verbose=0)

    probabilities = prediction[0].tolist()

    confidence = float(np.max(prediction)*100)

    index = np.argmax(prediction)

    disease = CLASS_NAMES[index]

    recommendation = RECOMMENDATIONS[disease]

    heatmap = make_gradcam_heatmap(processed)

    overlay = overlay_heatmap(image,heatmap)

    gradcam_base64 = image_to_base64(overlay)

    del processed
    del prediction
    del heatmap
    del overlay
    gc.collect()

    return{

        "disease":disease,

        "confidence":confidence,

        "recommendation":recommendation,

        "gradcam":gradcam_base64,

        "probabilities": probabilities


    }


def is_greenish(image, threshold=0.25):
    """
    Simple heuristic: check if a meaningful portion of the image
    is green-toned, which most leaf photos are, even when a
    generic ImageNet classifier misreads the texture.
    """
    img_array = np.array(image.convert("RGB")).astype(np.float32)
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    green_mask = (g > r) & (g > b * 0.9) & (g > 40)
    green_ratio = np.mean(green_mask)
    return green_ratio > threshold


def validate_leaf(image):
    """
    Validate whether the uploaded image is likely to be a tomato leaf.
    MobileNet is loaded only when needed to reduce startup memory.
    """
    try:
        from tensorflow.keras.applications.mobilenet_v2 import (
            MobileNetV2,
            preprocess_input,
            decode_predictions
        )

        leaf_validator = MobileNetV2(
            weights="imagenet",
            include_top=True
        )

        img = image.resize((224, 224))
        img_array = np.array(img)

        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]

        img_array = np.expand_dims(
            img_array.astype(np.float32),
            axis=0
        )
        img_array = preprocess_input(img_array)

        predictions = leaf_validator.predict(
            img_array,
            verbose=0
        )

        decoded = decode_predictions(predictions, top=5)[0]
        logging.info(f"Leaf validation predictions: {decoded}")

        keywords = [
            "leaf",
            "tomato",
            "cabbage",
            "broccoli",
            "cauliflower",
            "corn"
        ]

        is_leaf_keyword = any(
            keyword in label.lower() and confidence > 0.03
            for (_, label, confidence) in decoded
            for keyword in keywords
        )

        is_leaf_color = is_greenish(image)

        is_leaf = is_leaf_keyword or is_leaf_color

        logging.info(
            f"is_leaf_keyword={is_leaf_keyword}, is_leaf_color={is_leaf_color}, final={is_leaf}"
        )

        del predictions
        del leaf_validator
        gc.collect()

        return is_leaf

    except Exception as e:
        print("Validation Error:", e)
        return False




#########################################################
# Rule-Based XAI Explanation
#########################################################

def generate_xai(disease):

    explanations = {

        "Tomato_Bacterial_spot":[

            "Grad-CAM highlights dark bacterial lesion regions.",

            "The CNN focuses on infected leaf tissue.",

            "Highlighted regions are consistent with bacterial spot."

        ],

        "Tomato_Early_blight":[

            "Grad-CAM highlights brown necrotic lesions.",

            "The CNN focuses on damaged tissue.",

            "These regions indicate Early Blight."

        ],

        "Tomato_Late_blight":[

            "Grad-CAM highlights irregular dark lesions.",

            "The CNN focuses on infected tissue.",

            "This pattern matches Late Blight."

        ],

        "Tomato_Leaf_Mold":[

            "Grad-CAM highlights mold infected regions.",

            "CNN focuses on fungal affected tissue.",

            "These regions indicate Leaf Mold."

        ],

        "Tomato_Septoria_leaf_spot":[

            "Grad-CAM highlights circular lesions.",

            "CNN identifies infected tissue.",

            "These symptoms match Septoria Leaf Spot."

        ],

        "Tomato_Spider_mites_Two_spotted_spider_mite":[

            "Grad-CAM highlights damaged leaf tissue.",

            "CNN focuses on mite feeding damage.",

            "Highlighted regions indicate Spider Mites."

        ],

        "Tomato_Target_Spot":[

            "Grad-CAM highlights target-like lesions.",

            "CNN focuses on infected regions.",

            "These symptoms indicate Target Spot."

        ],

        "Tomato_Tomato_YellowLeaf_Curl_Virus":[

            "Grad-CAM highlights curled yellow regions.",

            "CNN focuses on abnormal leaf deformation.",

            "Highlighted regions indicate Yellow Leaf Curl Virus."

        ],

        "Tomato_Tomato_mosaic_virus":[

            "Grad-CAM highlights mosaic discoloration.",

            "CNN focuses on abnormal pigmentation.",

            "These symptoms indicate Tomato Mosaic Virus."

        ],

        "Tomato_healthy":[

            "Grad-CAM activation is evenly distributed.",

            "No strong disease region detected.",

            "Leaf characteristics appear healthy."

        ]

    }

    return explanations[disease]

#########################################################
# Home
#########################################################

@app.route("/",methods=["GET"])

def home():

    return jsonify({

         "application": APP_NAME,

         "version": VERSION,

         "author": AUTHOR,

         "status": "Running"

    })

#########################################################
# Health Check
#########################################################

@app.route("/health", methods=["GET"])
def health():

    return jsonify({

        "status": "Healthy",

        "model_loaded": model is not None,

        "gradcam": LAST_CONV_LAYER,

        "classes": len(CLASS_NAMES)

    })

#########################################################
# Prediction API
#########################################################

#########################################################
# Prediction API
#########################################################

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Check uploaded image
        if "image" not in request.files:
            return jsonify({
                "error": "No image uploaded."
            }), 400

        image_file = request.files["image"]

        logging.info(f"Image Uploaded: {image_file.filename}")

        # Check file extension
        filename = image_file.filename.lower()
        allowed = (".jpg", ".jpeg", ".png")

        if not filename.endswith(allowed):
            return jsonify({
                "error": "Only JPG, JPEG and PNG images are allowed."
            }), 400

        # Read image
        image = Image.open(image_file).convert("RGB")

        ####################################################
        # MobileNetV2 Validation
        ####################################################

        if not validate_leaf(image):

            logging.warning("Invalid image detected by MobileNetV2.")


            return jsonify({
               "status": "invalid_image",
               "error": "Please upload a valid tomato leaf image."
            }), 400

        ####################################################
        # Disease Prediction
        ####################################################

        result = predict_disease(image)

        logging.info(
            f"Prediction: {result['disease']} | "
            f"{result['confidence']:.2f}%"
        )
    
        tf.keras.backend.clear_session()
        gc.collect()
        
        return jsonify({

            "disease": result["disease"],

            "confidence": round(result["confidence"], 2),

            "recommendation": result["recommendation"],

            "gradcam_base64": result["gradcam"],

            "probabilities": result["probabilities"],

            "xai": generate_xai(result["disease"])

        })

    except Exception as e:

        logging.exception("Prediction Failed")

        return jsonify({

            "error": str(e)

        }), 500

#########################################################
# Run Flask
#########################################################

if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
