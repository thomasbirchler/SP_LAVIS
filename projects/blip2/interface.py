import os
import socket

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


############## SOCKET ##############
def setup_socket():
    host = '192.168.1.204'
    port = 54321
    #  Create a server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(4)
    print("Waiting for a connection...")

    # Accept a client connection
    client_socket, addr = server_socket.accept()
    print("Connected by", addr)

    return client_socket


def receive_and_save_from_socket(client_socket):
    while True:
        file_data = b''
        file_name = b''

        # Receive the filename
        while True:
            data = client_socket.recv(1)
            if data == b'\0':
                break
            file_name += data
        # Decode the filename
        file_name = file_name.decode("utf-8")

        # Specify the saving location and file name
        save_location = "/home/bthomas/SP_LAVIS/inputs/socket/"
        file_path = os.path.join(save_location, file_name)

        # Receive and save the file based on its extension
        with open(file_path, 'wb') as file:
            buffer = b''
            while True:
                # Receive and save the text file
                text_data = client_socket.recv(1024)
                if not text_data:
                    break
                # Add the newly received data to the buffer
                buffer += text_data
                while b'FILE_END' in buffer:
                    # Split buffer at the first occurrence of 'FILE_END'
                    file_data, buffer = buffer.split(b'FILE_END', 1)
                    # Write the data to the file
                    # file_data = file_data.decode("utf-8")
                    file.write(file_data)
                    print(f"Text file {file_name} saved.")
                    return

            # elif file_extension in ('jpg', 'jpeg'):
            #     Receive the entire image as a single block of binary data
                # image_data = b""
                # while True:
                #     chunk = client_socket.recv(1024)
                #     if not chunk:
                #         break
                #     image_data += chunk
                #     if data == b'FILE_END':
                #         break
                # Write the entire binary image data to the file
                # file.write(image_data)
                # print(f"Image file {file_name} saved.")
            # else:
            #     print(f"Unsupported file extension: {file_extension}")


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
    client_socket = setup_socket()
    while True:
        print("Waiting for new input.")
        receive_and_save_from_socket(client_socket)

    # model, vis_processors = load_model_with_preprocessors()
    # print("Model loaded.")
    # for prompt_number in range(len(prompts)):
    #     for index in range(4):
    #         for direction in directions:
    #             print(f'index: {index}, direction: {directions[0]}')
    #             raw_image = get_image(index, direction)
    #             print(f'got raw image{index}_{direction}')
    # image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    # print(f"Starting caption generation for frame{index}_{direction}.")
    # caption = generate_caption(model, image, prompts[prompt_number])
    # save_caption(caption, index, direction, prompt_number)


if __name__ == "__main__":
    main()
