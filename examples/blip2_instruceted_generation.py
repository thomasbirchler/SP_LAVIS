import sys
import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess


def main():
    if 'google.colab' in sys.modules:
        print('Running in Colab.')
        # !pip install salesforce-lavis

    # Load an example image
    # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
    img_url = "../docs/_static/merlion.png"
    # img_url = "https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.theseverngroup.com%2Finindoor-environmental" \
    #           "-quality-ieq%2F&psig=AOvVaw2cRXp_ptfHr-2ReR5KgotQ&ust=1700817698430000&source=images&cd=vfe&opi" \
    #           "=89978449&ved=0CBEQjRxqFwoTCMDH7ufl2YIDFQAAAAAdAAAAABAF"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    # raw_image = Image.open(img_url).convert('RGB')


    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # Load pretrained/finetuned BLIP2 captioning model
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
    )

    # prepare the image as model input using the associated processors
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    # generate caption using beam search
    generated_captions_beam_search = model.generate({"image": image})

    # generate multiple captions using nucleus sampling
    # due to the non-deterministic nature of nucleus sampling, you may get different captions.
    generated_captions_nucleus_sampling = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=3)

    # Print the generated captions
    print("Generated Captions (Beam Search):")
    for caption in generated_captions_beam_search:
        print(caption)

    print("\nGenerated Captions (Nucleus Sampling):")
    for caption in generated_captions_nucleus_sampling:
        print(caption)

    # instructed zero-shot vision-to-language generation
    model.generate({"image": image, "prompt": "Question: which city is this? Answer:"})

    model.generate({
        "image": image,
        "prompt": "Question: which city is this? Answer: singapore. Question: why?"})

    context = [
        ("which city is this?", "singapore"),
        ("why?", "it has a statue of a merlion"),
    ]
    question = "where is the name merlion coming from?"
    template = "Question: {} Answer: {}."

    prompt = " ".join([template.format(context[i][0], context[i][1]) for i in
                       range(len(context))]) + " Question: " + question + " Answer:"

    print(prompt)

    model.generate(
        {
            "image": image,
            "prompt": prompt
        },
        use_nucleus_sampling=False,
    )


if __name__ == "__main__":
    main()
