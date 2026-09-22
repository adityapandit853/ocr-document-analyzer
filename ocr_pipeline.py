import os
import re
import cv2
import pytesseract
import easyocr
import pandas as pd
import numpy as np

from PIL import Image
from pytesseract import Output


# CONFIGURATION

INPUT_FOLDER = "sample_images"
OUTPUT_FOLDER = "output"

# Confidence threshold for fallback
CONFIDENCE_THRESHOLD = 60.0

# Tesseract path for Windows
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Supported image extensions
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
)



# SETUP

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Configure Tesseract
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    print("\nWARNING:")
    print("Tesseract executable was not found at:")
    print(TESSERACT_PATH)
    print("If Tesseract is installed somewhere else,")
    print("change TESSERACT_PATH in this script.\n")



# INITIALIZE EASYOCR

print("=" * 70)
print("INITIALIZING EASYOCR")
print("=" * 70)

try:
    reader = easyocr.Reader(
        ["en"],
        gpu=False
    )

    print("EasyOCR initialized successfully.")

except Exception as e:
    print("EasyOCR initialization failed:")
    print(e)
    reader = None



# TASK 1: IMAGE INFORMATION


def get_image_information(image_path):

    try:

        pil_image = Image.open(image_path)

        width, height = pil_image.size
        image_format = pil_image.format

        return {
            "width": width,
            "height": height,
            "format": image_format
        }

    except Exception as e:

        return {
            "width": 0,
            "height": 0,
            "format": "Unknown"
        }



# TASK 3: IMAGE PREPROCESSING


def preprocess_image(image):

    """
    Performs:
    1. Grayscale conversion
    2. Noise removal
    3. Thresholding
    """

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Noise removal
    denoised = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Adaptive thresholding
    threshold = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return threshold



# TASK 2: BASIC TESSERACT OCR


def tesseract_ocr(image):

    try:

        text = pytesseract.image_to_string(
            image,
            config="--psm 6"
        )

        return text.strip()

    except Exception as e:

        print("Tesseract OCR error:", e)

        return ""



# TASK 6: TESSERACT CONFIDENCE SCORE


def get_tesseract_confidence(image):

    try:

        data = pytesseract.image_to_data(
            image,
            output_type=Output.DICT,
            config="--psm 6"
        )

        confidence_values = []

        for conf in data["conf"]:

            try:

                conf_value = float(conf)

                if conf_value >= 0:
                    confidence_values.append(conf_value)

            except:
                pass

        if len(confidence_values) == 0:
            return 0.0

        average_confidence = (
            sum(confidence_values)
            / len(confidence_values)
        )

        return round(
            average_confidence,
            2
        )

    except Exception as e:

        print("Confidence calculation error:", e)

        return 0.0



# TASK 5: LOW CONFIDENCE / ANOMALY DETECTION

def detect_low_confidence(
    text,
    confidence,
    image
):

    reasons = []

    # Rule 1: confidence
    if confidence < CONFIDENCE_THRESHOLD:

        reasons.append(
            f"Low OCR confidence ({confidence:.2f}%)"
        )

    # Rule 2: very short text
    word_count = len(text.split())

    if word_count < 3:

        reasons.append(
            "Extracted text is very short"
        )

    # Rule 3: unusual characters
    if len(text) > 0:

        unusual_characters = re.findall(
            r"[^a-zA-Z0-9\s.,!?;:'\"()\-/]",
            text
        )

        unusual_ratio = (
            len(unusual_characters)
            / len(text)
        )

        if unusual_ratio > 0.20:

            reasons.append(
                "Too many unusual characters"
            )

    # Rule 4: image size vs text
    height, width = image.shape[:2]

    if width * height > 500000 and word_count < 5:

        reasons.append(
            "Large image but very little extracted text"
        )

    if len(reasons) > 0:

        return True, reasons

    return False, []



# TASK 4: EASYOCR

def easyocr_extract(image):

    if reader is None:

        return "", 0.0

    try:

        results = reader.readtext(
            image
        )

        extracted_words = []

        confidence_values = []

        for result in results:

            if len(result) >= 3:

                text = result[1]
                confidence = result[2]

                extracted_words.append(
                    text
                )

                confidence_values.append(
                    confidence * 100
                )

        final_text = " ".join(
            extracted_words
        )

        if len(confidence_values) > 0:

            average_confidence = (
                sum(confidence_values)
                / len(confidence_values)
            )

        else:

            average_confidence = 0.0

        return (
            final_text.strip(),
            round(
                average_confidence,
                2
            )
        )

    except Exception as e:

        print("EasyOCR error:", e)

        return "", 0.0



# TASK 7: SAVE PREPROCESSED IMAGE

def save_preprocessed_image(
    image,
    filename
):

    output_path = os.path.join(
        OUTPUT_FOLDER,
        "preprocessed_" + filename
    )

    cv2.imwrite(
        output_path,
        image
    )

    return output_path



# TASK 8: FALLBACK LOGIC


def perform_ocr_with_fallback(
    original_image,
    preprocessed_image
):

    # First attempt with Tesseract
    tesseract_text = tesseract_ocr(
        preprocessed_image
    )

    tesseract_confidence = (
        get_tesseract_confidence(
            preprocessed_image
        )
    )

    # Check whether Tesseract result is acceptable
    low_confidence, reasons = (
        detect_low_confidence(
            tesseract_text,
            tesseract_confidence,
            original_image
        )
    )

    # TESSERACT RESULT ACCEPTED

    if not low_confidence:

        return {
            "final_text": tesseract_text,
            "final_confidence": tesseract_confidence,
            "engine": "Tesseract",
            "fallback_used": False,
            "reasons": ""
        }


    # FALLBACK TO EASYOCR
    print(
        "  Tesseract confidence is low."
    )

    print(
        "  Trying EasyOCR..."
    )

    easy_text, easy_confidence = (
        easyocr_extract(
            original_image
        )
    )

    # If EasyOCR produces useful text
    if len(easy_text.strip()) > 0:

        return {
            "final_text": easy_text,
            "final_confidence": easy_confidence,
            "engine": "EasyOCR",
            "fallback_used": True,
            "reasons": "; ".join(reasons)
        }

    # If EasyOCR fails
    return {
        "final_text": tesseract_text,
        "final_confidence": tesseract_confidence,
        "engine": "Tesseract",
        "fallback_used": False,
        "reasons": "; ".join(reasons)
    }



# TASK 9: COMPLETE DOCUMENT PROCESSING FUNCTION


def process_document(image_path):

    filename = os.path.basename(
        image_path
    )

    print("\n")
    print("=" * 70)
    print("PROCESSING:", filename)
    print("=" * 70)
  
    # LOAD IMAGE

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "ERROR: Could not load image."
        )

        return None

    # IMAGE INFORMATION

    info = get_image_information(
        image_path
    )

    print(
        f"Image size: {info['width']} x {info['height']}"
    )

    print(
        f"Format: {info['format']}"
    )

    # BASIC TESSERACT

    print("\n--- BASIC TESSERACT OCR ---")

    basic_text = tesseract_ocr(
        image
    )

    print(
        basic_text[:500]
    )


    # PREPROCESS IMAGE

    print("\n--- PREPROCESSING IMAGE ---")

    processed_image = preprocess_image(
        image
    )

    save_preprocessed_image(
        processed_image,
        filename
    )


    # TESSERACT AFTER PREPROCESSING

    print(
        "\n--- TESSERACT AFTER PREPROCESSING ---"
    )

    processed_text = tesseract_ocr(
        processed_image
    )

    processed_confidence = (
        get_tesseract_confidence(
            processed_image
        )
    )

    print(
        processed_text[:500]
    )

    print(
        f"\nTesseract confidence: "
        f"{processed_confidence:.2f}%"
    )

  
    # TASK 4: EASY OCR COMPARISON
 

    print("\n--- EASYOCR ---")

    easy_text, easy_confidence = (
        easyocr_extract(
            image
        )
    )

    print(
        easy_text[:500]
    )

    print(
        f"\nEasyOCR confidence: "
        f"{easy_confidence:.2f}%"
    )


    # LOW CONFIDENCE CHECK

    low_confidence, reasons = (
        detect_low_confidence(
            processed_text,
            processed_confidence,
            image
        )
    )

    print(
        "\n--- CONFIDENCE ANALYSIS ---"
    )

    if low_confidence:

        print(
            "STATUS: LOW CONFIDENCE"
        )

        for reason in reasons:

            print(
                "Reason:",
                reason
            )

    else:

        print(
            "STATUS: GOOD OCR RESULT"
        )

  
    # FALLBACK

    final_result = (
        perform_ocr_with_fallback(
            image,
            processed_image
        )
    )

    print(
        "\n--- FINAL RESULT ---"
    )

    print(
        "Engine:",
        final_result["engine"]
    )

    print(
        "Fallback used:",
        final_result["fallback_used"]
    )

    print(
        "Confidence:",
        final_result["final_confidence"]
    )

    print(
        "Final text:"
    )

    print(
        final_result["final_text"][:1000]
    )

  
    # RETURN RESULT
  
    result = {

        "filename": filename,

        "width": info["width"],

        "height": info["height"],

        "format": info["format"],

        "tesseract_text": processed_text,

        "tesseract_word_count":
            len(processed_text.split()),

        "tesseract_confidence":
            processed_confidence,

        "easyocr_text":
            easy_text,

        "easyocr_word_count":
            len(easy_text.split()),

        "easyocr_confidence":
            easy_confidence,

        "final_engine":
            final_result["engine"],

        "final_text":
            final_result["final_text"],

        "final_word_count":
            len(
                final_result[
                    "final_text"
                ].split()
            ),

        "final_confidence":
            final_result[
                "final_confidence"
            ],

        "fallback_used":
            final_result[
                "fallback_used"
            ],

        "status":
            (
                "LOW CONFIDENCE"
                if low_confidence
                else "GOOD"
            ),

        "reasons":
            final_result[
                "reasons"
            ]
    }

    return result



# TASK 7: PROCESS ALL DOCUMENTS


def process_all_documents():

    print("\n")
    print("#" * 70)
    print("DOCUMENT ANALYZER - OCR PROCESSING")
    print("#" * 70)

    # Check input directory
    if not os.path.exists(
        INPUT_FOLDER
    ):

        print(
            f"\nERROR: Folder '{INPUT_FOLDER}' does not exist."
        )

        print(
            "Create the folder and put your sample images inside it."
        )

        return

    # Find images
    image_files = []

    for filename in os.listdir(
        INPUT_FOLDER
    ):

        if filename.lower().endswith(
            IMAGE_EXTENSIONS
        ):

            image_files.append(
                filename
            )

    if len(image_files) == 0:

        print(
            "\nNo images found."
        )

        print(
            "Put at least 5 images inside:"
        )

        print(
            INPUT_FOLDER
        )

        return

    print(
        f"\nFound {len(image_files)} image(s)."
    )

    all_results = []

  
    # PROCESS EACH IMAGE


    for filename in image_files:

        image_path = os.path.join(
            INPUT_FOLDER,
            filename
        )

        result = process_document(
            image_path
        )

        if result is not None:

            all_results.append(
                result
            )

  
    # CREATE DATAFRAME
  
    if len(all_results) == 0:

        print(
            "No documents were processed."
        )

        return

    df = pd.DataFrame(
        all_results
    )


    # SAVE FULL CSV REPORT

    csv_path = os.path.join(
        OUTPUT_FOLDER,
        "ocr_report.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8"
    )

    print("\n")
    print("#" * 70)
    print("OCR REPORT")
    print("#" * 70)

    # Print important columns
    display_columns = [

        "filename",

        "final_engine",

        "final_word_count",

        "final_confidence",

        "fallback_used",

        "status"
    ]

    print(
        df[display_columns].to_string(
            index=False
        )
    )


    # SAVE FINAL TEXT REPORT
  
    text_report_path = os.path.join(
        OUTPUT_FOLDER,
        "final_text_report.txt"
    )

    with open(
        text_report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "DOCUMENT ANALYZER - OCR REPORT\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        for result in all_results:

            file.write(
                f"FILE: {result['filename']}\n"
            )

            file.write(
                f"ENGINE: {result['final_engine']}\n"
            )

            file.write(
                f"CONFIDENCE: "
                f"{result['final_confidence']:.2f}%\n"
            )

            file.write(
                f"WORD COUNT: "
                f"{result['final_word_count']}\n"
            )

            file.write(
                f"FALLBACK USED: "
                f"{result['fallback_used']}\n"
            )

            file.write(
                f"STATUS: "
                f"{result['status']}\n"
            )

            file.write(
                "\nEXTRACTED TEXT:\n"
            )

            file.write(
                result["final_text"]
            )

            file.write(
                "\n\n"
            )

            file.write(
                "-" * 70
                + "\n\n"
            )

    # FINAL SUMMARY

    print("\n")
    print("#" * 70)
    print("PROCESSING COMPLETED")
    print("#" * 70)

    print(
        f"\nDocuments processed: "
        f"{len(all_results)}"
    )

    print(
        f"CSV report: "
        f"{csv_path}"
    )

    print(
        f"Text report: "
        f"{text_report_path}"
    )

    print(
        f"\nPreprocessed images saved in:"
        f"\n{OUTPUT_FOLDER}"
    )

    print(
        "\nAll OCR tasks completed successfully."
    )



# MAIN

if __name__ == "__main__":

    process_all_documents()