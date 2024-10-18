# Generating and Evaluating Multimodal Embeddings with [BridgeTower](https://huggingface.co/docs/transformers/en/model_doc/bridgetower) Model
This repository contains a Python script for generating and evaluating multimodal embeddings from image-caption pairs using the BridgeTower model from Hugging Face's `transformers` library.

- Embedding Generation: 

The script uses the BridgeTower model from Hugging Face to generate embeddings from both images and text. The processor takes care of handling the raw images and captions.

- Evaluation: 

The script computes cosine similarity and Euclidean distance between the embeddings, helping to measure how similar the images and captions are in their joint embedding space.

- Model Details

BridgeTower is a cutting-edge multimodal model that aligns images and text into a shared representation space. It's particularly effective for image-text retrieval and other vision-language tasks. This script leverages the bridgetower-large-itm-mlm-itc variant, which is fine-tuned for various vision-language tasks such as image captioning, question-answering, and embedding generation.

## Models Used
**BridgeTower Model**: This script uses the `BridgeTower/bridgetower-large-itm-mlm-itc` model from Hugging Face. BridgeTower is a state-of-the-art vision-language model that generates joint embeddings from both visual (image) and textual (caption) inputs. The output embeddings are 768-dimensional vectors that can be used for tasks like image-text retrieval, similarity matching, multimodal RAG and more.

## Dependencies
Before running the script, make sure you have the following libraries installed:

```bash
pip install torch transformers Pillow numpy opencv-python requests
```

## How to Run

1. Prepare JSON File: 

You need to provide a JSON file containing the image metadata (image URL, caption, and local file path). Here's an example format for the JSON file (let's call it image_data.json):

```json
[
  {
    "flickr_url": "",
    "caption": "",
    "image_path": "./name.jpg"
  },
  {
    "flickr_url": "",
    "caption": "",
    "image_path": "./.jpg"
  },
  {
    "flickr_url": "",
    "caption": "",
    "image_path": "./.jpg"
  }
]
```
Make sure the JSON file is structured properly with URLs for each image and corresponding captions.

2. Run the Script

**Generate Embeddings**

To generate the embeddings from the image-caption pairs, run the following command:

```bash
python multimodal-bridgetower-embedding.py --json-file image_data.json --generate-embeddings
```
This command will:

Download the images from the URLs in the JSON file (if not already present).
Generate 768-dimensional embeddings using the BridgeTower model.
Display the first 10 values and length of each embedding.
Save the embeddings in embeddings.json.

**Evaluate Embeddings**

Once you have the embeddings generated, you can compare them based on cosine similarity and Euclidean distance:

```bash
python script.py --json-file image_data.json --evaluate-embedding
```

This command will:

Load the embeddings from embeddings.json.
Compute the cosine similarity and Euclidean distance between the embeddings of the first, second, and third images.
Display the results in the terminal.

**Reference**: This code is adopted & inspired from the deeplearning.ai courses


