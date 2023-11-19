import torch
from PIL import Image
from lavis.models import load_model_and_preprocess


# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


def get_image():
    # load sample image
    raw_image = Image.open("../../docs/_static/merlion.png").convert("RGB")
    # display(raw_image.resize((596, 437)))
    return raw_image


def load_model_with_preprocessors(raw_image):
    # loads BLIP-2 pre-trained model
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True,
                                                         device=device)
    # prepare the image
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    return image


def generate_response(image, output_filename="image_caption.txt"):
    response_text = model.generate({"image": image, "prompt": "Question: which city is this? Answer:"})
    # 'singapore'

    # Save the response to a text file
    with open(output_filename, "w") as file:
        file.write(response_text)



def save_response(output_filename="response.txt"):
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

def explain_answer():
    model.generate({
        "image": image,
        "prompt": "Question: which city is this? Answer: singapore. Question: why?"})
    # 'it has a statue of a merlion'


def main():
    print("starting with the program")
    raw_image = get_image()
    image = load_model_with_preprocessors(raw_image)
    generate_response()
    save_response()
    # explain_answer()
    pass


if __name__ == "__main__":
    main()
