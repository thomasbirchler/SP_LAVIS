import os
import socket
import time

import torch
from PIL import Image
from lavis.models import load_model_and_preprocess



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


def receive_data_from_socket(client_socket):
    file_name = b''

    # Receive the filename
    while True:
        data = client_socket.recv(1)
        if data == b'\0':
            break
        file_name += data
    # Decode the filename
    file_name = file_name.decode("utf-8")

    if file_name == "exit":
        return True

    # Specify the saving location and file name
    # save_location = "/home/bthomas/SP_LAVIS/inputs/frames/"
    # file_path = os.path.join(save_location, file_name)
    file_path = "../../inputs/frames/" + file_name

    # Receive and save the jpg file
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
                file.write(file_data)
                print(f"File {file_name} saved from socket.")
                return False


def send_data_to_socket(frame, direction, client_socket):
    file_name = f"caption{frame:04}_{direction}.txt"
    file_path = "../../output/" + file_name
    # Send the filename first
    client_socket.send(file_name.encode("utf-8"))

    # Send a delimiter to separate filename and content
    client_socket.send(b'\0')
    with open(file_path, 'rb') as text_file:
        # Send file in batches
        text_data = text_file.read(1024)
        while text_data:
            client_socket.send(text_data)
            text_data = text_file.read(1024)
    # Send a marker to indicate the end of the file
    client_socket.send(b'FILE_END')
    print(f"File {file_name} sent.")



############## Captioning ##############

def get_image(image_index, direction):
    image_filename = f'../../inputs/frames/frame{image_index:04d}_{direction}.jpg'
    while True:
        if os.path.exists(image_filename):
            return Image.open(image_filename)


def load_model_with_preprocessors(device):
    # loads BLIP-2 pre-trained model
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True,
                                                         device=device)
    return model, vis_processors


def generate_caption(model, image):
    # Prompt used to create image caption
    prompt = "Describe the image in detail."

    # Generate image caption with prompt and image
    response_text = model.generate({"image": image, "prompt": prompt})

    return response_text


def save_caption(caption, index, direction):
    # Get the current working directory
    current_directory = os.getcwd()
    # Specify the relative path to the parent directory and format
    relative_path = "../../output/caption" + f'{index:04}_{direction}.txt'
    # Construct the full output file path
    output_file = os.path.join(current_directory, relative_path)
    # Transform list into one string
    caption = ''.join(caption)
    # Save the response to a text file
    with open(output_file, "w") as file:
        file.write(caption)
    print(f"Saved caption of frame_{index:04}.")



############## Main ##############

def main():
    # Keep track of frame/step
    frame = 0
    # Cease program if target object is detected
    target_object_found = False
    # All directions
    directions = ["left", "front", "right"]
    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    print("Loading model.")
    model, vis_processors = load_model_with_preprocessors(device)

    client_socket = setup_socket()

    while True:
        print("Waiting for new input.")
        for i in range(3):
            target_object_found = receive_data_from_socket(client_socket)
            if target_object_found:
                break

        if target_object_found:
            break

        for direction in directions:
            raw_image = get_image(frame, direction)
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            print(f"Starting caption generation for frame{frame:04}_{direction}.")
            caption = generate_caption(model, image)
            save_caption(caption, frame, direction)
            send_data_to_socket(frame, direction, client_socket)

        frame += 1

    client_socket.close()


if __name__ == "__main__":
    main()
