import os

import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# set different prompts for testing
prompts = ["Question: Describe the image in detail.",
           "Question: What is visible in the image.",
           "Command: Describe the image in detail.",
           "Describe the image in detail.",
           "Describe this picture to a blind person.",
           "Describe everything in the image.",
           "Describe everything in the image and the context of it.",
           "List all objects in the image."]

# All directions
directions = ["left", "front", "right"]


def get_image(image_index, direction):
    image_filename = f'../../inputs/frames/frame{image_index:04d}_{direction}.jpg'
    print(image_filename)
    while True:
        if os.path.exists(image_filename):
            # if os.path.exists(image_filename + f'{directions[1]}.jpg'):
            #     if os.path.exists(image_filename + f'{directions[2]}.jpg'):
            #         image_left = Image.open(image_filename+f'{directions[0]}.jpg')
            #         image_front = Image.open(image_filename+f'{directions[1]}.jpg')
            #         image_right = Image.open(image_filename+f'{directions[2]}.jpg')
            #         return image_left, image_front, image_right, image_index
            return Image.open(image_filename)


def load_model_with_preprocessors():
    # loads BLIP-2 pre-trained model
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True,
                                                         device=device)
    return model, vis_processors


def generate_caption(model, image, prompt):
    # prompt = "Question: Describe everything in this picture. Answer:"
    response_text = model.generate({"image": image, "prompt": prompt})
    return response_text[0]


def save_caption(caption, index, direction, prompt_number):
    # Get the current working directory
    current_directory = os.getcwd()
    # Specify the relative path to the parent directory and format
    relative_path = "../../output/caption" + f'{index:04}_{direction}_prompt{prompt_number}.txt'
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
    for prompt_number in range(len(prompts)):
        for index in range(4):
            for direction in directions:
                print(f'index: {index}, direction: {directions[0]}')
                raw_image = get_image(index, direction)
                print(f'got raw image{index}_{direction}')
                image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                print(f"Starting caption generation for frame{index}_{direction}.")
                caption = generate_caption(model, image, prompts[prompt_number])
                save_caption(caption, index, direction, prompt_number)


if __name__ == "__main__":
    main()




