import requests
import json
import argparse
import base64
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from urllib.request import urlopen
from transformers import BridgeTowerProcessor, BridgeTowerModel
from numpy.linalg import norm
import cv2

# Function to encode image from a path or URL into base64
def encode_image_from_path_or_url(image_path_or_url):
    try:
        f = urlopen(image_path_or_url)
        return base64.b64encode(requests.get(image_path_or_url).content).decode('utf-8')
    except:
        with open(image_path_or_url, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

# Function to generate joint embedding using BridgeTower for image and text
def bt_embedding_from_bridge_tower(prompt, base64_image):
    image_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data))

    processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
    model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")

#     processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base")
#     model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base")
    
#     processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
#     model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")

    inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.pooler_output
    return embedding.squeeze().tolist()

# Function to compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    similarity = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    return similarity

# Function to compute Euclidean distance between two vectors
def euclidean_distance(vec1, vec2):
    return cv2.norm(np.array(vec1), np.array(vec2), cv2.NORM_L2)

# Main function
def main(args):
    # Load the JSON file containing the image metadata
    with open(args.json_file, 'r') as f:
        imgs = json.load(f)

    # Generate embeddings if flag is set
    if args.generate_embeddings:
        embeddings = []
        for img in imgs:
            encoded_image = encode_image_from_path_or_url(img['flickr_url'])
            embedding = bt_embedding_from_bridge_tower(img['caption'], encoded_image)
            embeddings.append(embedding)
            print(f"Caption: {img['caption']}")
            print(f"Multimodal embedding (first 10 values): {embedding[:10]}")
            print(f"Embedding length: {len(embedding)}")
            print("-" * 50)
        # Save embeddings to a file
        with open('embeddings.json', 'w') as f:
            json.dump(embeddings, f)
    # Evaluate embeddings if flag is set
    if args.evaluate_embedding:
        # Load embeddings from the file
        with open('embeddings.json', 'r') as f:
            embeddings = json.load(f)
        
        ex1_embed = np.array(embeddings[0])
        ex2_embed = np.array(embeddings[1])
        ex3_embed = np.array(embeddings[2])
        
        # Cosine similarities
        sim_ex1_ex2 = cosine_similarity(ex1_embed, ex2_embed)
        sim_ex1_ex3 = cosine_similarity(ex1_embed, ex3_embed)
        
        print(f"Cosine similarity between example 1 and 2: {sim_ex1_ex2}")
        print(f"Cosine similarity between example 1 and 3: {sim_ex1_ex3}")
        
        # Euclidean distances
        dist_ex1_ex2 = euclidean_distance(ex1_embed, ex2_embed)
        dist_ex1_ex3 = euclidean_distance(ex1_embed, ex3_embed)
        
        print(f"Euclidean distance between example 1 and 2: {dist_ex1_ex2}")
        print(f"Euclidean distance between example 1 and 3: {dist_ex1_ex3}")

# Define arguments and flags
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and Evaluate Embeddings")
    parser.add_argument('--json-file', type=str, required=True, help='Path to the JSON file containing image metadata')
    parser.add_argument('--generate-embeddings', action='store_true', help='Flag to generate embeddings')
    parser.add_argument('--evaluate-embedding', action='store_true', help='Flag to evaluate embeddings')
    
    args = parser.parse_args()
    main(args)
