import os

import torch
from PIL import Image
from lavis.models import load_model_and_preprocess


# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# Todo: get number from read image
response_text_frame = 0


def get_image():
    # load sample image
    # raw_image = Image.open("../../docs/_static/merlion.png").convert("RGB")
    raw_image = Image.open("../../frames/frame0007.jpg").convert("RGB")
    # display(raw_image.resize((596, 437)))

    return raw_image


def load_model_with_preprocessors(raw_image):
    # loads BLIP-2 pre-trained model
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True,
                                                         device=device)
    # prepare the image
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    return model, image


def generate_response(model, image, output_filename="image_caption.txt"):
    prompt = "Question: what can you see in the image? Answer:"
    response_text = model.generate({"image": image, "prompt": prompt})
    # 'singapore'
    return response_text
    # Save the response to a text file
    # with open(output_filename, "w") as file:
    #     file.write(response_text)


def save_response(response_text):
    global response_text_frame
    # Get the current working directory
    current_directory = os.getcwd()
    # Specify the relative path to the parent directory and format
    relative_path = "../../responses/response" + f'{response_text_frame:04}' + ".txt"
    # Construct the full output file path
    output_file = os.path.join(current_directory, relative_path)
    # Save the response to a text file
    with open(output_file, "w") as file:
        file.write(response_text[0])
        response_text_frame += 1


def generate_response2(model, image, output_filename="response.txt"):
    # Define device here
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    response = model.generate({"image": image, "prompt": "Question: which city is this? Answer:"})
    response_text = response[0]['generated_text']

    # Save the response to a text file
    with open(output_filename, "w") as file:
        file.write(response_text)

    # Print the response to the console
    print("Response:", response_text)

    # 'singapore'


def explain_answer(model, image):
    model.generate({
        "image": image,
        "prompt": "Question: which city is this? Answer: singapore. Question: why?"})
    # 'it has a statue of a merlion'


def main():
    print("starting with the program")
    raw_image = get_image()
    model, image = load_model_with_preprocessors(raw_image)
    while True:
        response_text = generate_response(model, image)
        # response_text = "a man on a unicycle"

        save_response(response_text)

    # explain_answer()
    pass


if __name__ == "__main__":
    main()
