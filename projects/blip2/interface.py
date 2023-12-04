import os

import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


def get_image(image_index):
    direction = ["left", "front", "right"]
    image_filename = f'../../inputs/frames/frame{image_index:04d}_'
    while True:
        file_path = image_filename + f'{direction[0]}.jpg'
        if os.path.exists(file_path):
            if os.path.exists(image_filename + f'{direction[1]}.jpg'):
                if os.path.exists(image_filename + f'{direction[2]}.jpg'):
                    image_left = Image.open(image_filename+f'{direction[0]}.jpg')
                    image_front = Image.open(image_filename+f'{direction[1]}.jpg')
                    image_right = Image.open(image_filename+f'{direction[2]}.jpg')
                    return image_left, image_front, image_right, image_index


def load_model_with_preprocessors():
    # loads BLIP-2 pre-trained model
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True,
                                                         device=device)
    return model, vis_processors


def generate_caption(model, image):
    prompt = "Question: Describe everything in this picture. Answer:"
    response_text = model.generate({"image": image, "prompt": prompt})
    return response_text[0]


def save_caption(caption, index, direction="left"):
    # Get the current working directory
    current_directory = os.getcwd()
    # Specify the relative path to the parent directory and format
    relative_path = "../../output/caption" + f'{index:04}_{direction}.txt'
    # Construct the full output file path
    output_file = os.path.join(current_directory, relative_path)
    # Save the response to a text file
    with open(output_file, "w") as file:
        file.write(caption)
    print(f"Saved caption of frame_{index} in output folder.")


def main():
    index = 0
    model, vis_processors = load_model_with_preprocessors()
    print("Model loaded.")
    while True:
        raw_image_left, raw_image_front, raw_image_right, index = get_image(index)
        print("got raw image")
        image_left = vis_processors["eval"](raw_image_left).unsqueeze(0).to(device)
        # image_front = vis_processors["eval"](raw_image_front).unsqueeze(0).to(device)
        # image_right = vis_processors["eval"](raw_image_right).unsqueeze(0).to(device)
        print("Starting caption generation.")
        caption = generate_caption(model, image_left)
        print("before save_caption")
        save_caption(caption, index)
        index += 1


if __name__ == "__main__":
    main()




