import tensorflow as tf
import numpy as np
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="out_relu"):
    # sub-model MobileNetV2 
    base_model = model.get_layer("mobilenetv2_1.00_224")
    
    grad_model = tf.keras.models.Model(
        inputs=base_model.inputs,
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )

    # rekam aktivasi dan gradien
    with tf.GradientTape() as tape:
        # normalisasi
        x = model.layers[1](img_array) 
        conv_outputs, base_out = grad_model(x)
        
        preds = model.layers[3](base_out)
        preds = model.layers[4](preds)
        preds = model.layers[5](preds)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(orig_img, heatmap, alpha=0.4):
    # ukuran heatmap 
    heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    superimposed = np.uint8(heatmap_colored * alpha + orig_img * (1 - alpha))
    return superimposed