import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
from flask import Flask, render_template, request, send_file
from io import BytesIO
import os
import numpy as np

app = Flask(__name__)

# Define the path for the generated images folder
GENERATED_IMAGES_FOLDER = os.path.join(app.root_path, 'generated_images')

# Ensure the folder exists, create if not
os.makedirs(GENERATED_IMAGES_FOLDER, exist_ok=True)

DEVICE = "cpu"

ALLOW_CUDA = False
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
model_file = "../data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)


"""
## TEXT TO IMAGE

# prompt = "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
prompt = "Generate a calming painting consisting of doraemon , mandala in background with a turquoise theme"
uncond_prompt = ""  # Also known as negative prompt
do_cfg = True
cfg_scale = 8  # min: 1, max: 14

## IMAGE TO IMAGE

input_image = None
# Comment to disable image to image
image_path = "../images/dog.jpg"
# input_image = Image.open(image_path)
# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
strength = 0.9

## SAMPLER

sampler = "ddpm"
num_inference_steps = 50
seed = 42

output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer,
)

# Combine the input image and the output image into a single image.
Image.fromarray(output_image)

"""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    prompt = request.form['prompt']
    uncond_prompt = request.form['uncond_prompt']
    
    # Handle file upload
    input_image = None
    if 'input_image' in request.files:
        input_image = Image.open(request.files['input_image'])

    strength = 0.9
    do_cfg = True
    cfg_scale = 8
    sampler = "ddpm"
    num_inference_steps = 50
    seed = 42
    """
    output_image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        seed=seed,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

     # Convert the NumPy array to a PIL Image object
    output_image_pil = Image.fromarray(np.uint8(output_image))

    # Save the output image to the generated_images folder
    output_image_path = os.path.join(GENERATED_IMAGES_FOLDER, 'generated_image.png')
    output_image_pil.save(output_image_path)

    # Save the output image to a BytesIO object
    output_buffer = BytesIO()
    output_image_pil.save(output_buffer, format='PNG')
    output_buffer.seek(0)

    # Return the output image as a file to be displayed in the browser
    return send_file(output_buffer, mimetype='image/png')
    """
    image_path = 'image.jpg'
    return send_file(image_path, mimetype='image/jpg')

if __name__ == '__main__':
    app.run(debug=True)