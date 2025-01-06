import fitz  # PyMuPDF
import pandas as pd
import json
import io
from PIL import Image
import numpy as np
from typing import Optional


def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df


def load_txt(file_path):
    with open(file_path, "r") as f:
        text = f.read()
    df = pd.DataFrame({"text": [text]})
    return df


def load_excel(file_path):
    df = pd.read_excel(file_path)
    return df


def load_image(file_path):
    img = Image.open(file_path)
    img_array = np.array(img)
    df = pd.DataFrame(
        {
            "image_array": [img_array.tobytes()],
            "shape": [img_array.shape],
            "dtype": [img_array.dtype.str],
        }
    )
    return df


def load_pdf(file_path):
    pdf_document = fitz.open(file_path)
    texts = []
    images = []

    for page_num, page in enumerate(pdf_document):
        # Extract text
        text = page.get_text()
        texts.append({"page": page_num + 1, "content": text})

        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]

            # Convert image to numpy array
            image = Image.open(io.BytesIO(image_bytes))
            img_array = np.array(image)

            images.append(
                {
                    "page": page_num + 1,
                    "index": img_index + 1,
                    "array": img_array.tobytes(),
                    "shape": img_array.shape,
                    "dtype": str(img_array.dtype),
                }
            )

    # Create DataFrame
    df = pd.DataFrame(
        {"texts": json.dumps(texts), "images": json.dumps(images)}, index=[0]
    )

    return df
