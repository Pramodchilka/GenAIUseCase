import os
import boto3
import mimetypes
from io import BytesIO
from transformers import pipeline
from PIL import Image, UnidentifiedImageError
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

# ======================
#  AWS Temporary Credentials Setup
# ======================
s3 = boto3.client(
    's3',
    aws_access_key_id='ASIASRGASLYJWJHFOJ7I',
    aws_secret_access_key='ltdcXEtRIvQtl2GFKsqtAUfbVDkxKEzSWMRDb+uQ',
    aws_session_token='IQoJb3JpZ2luX2VjEM7//////////wEaCmFwLXNvdXRoLTEiSDBGAiEAm8Fr9dsuqoHukgWSLFoGHs5YKjOaXYLCthrqpNy3gxkCIQDXbCTlSOjSgDGEJL7DsJr56uT8GI5Z6NUlxiy/agZGqSq3AwjI//////////8BEAAaDDE3NDM1MDAzMjQwMyIMDExo/kbYFStRcE+2KosDfHftbvsmwspRh5HsFE3o1tS/wqbBr85fp25WZwNP2Q/KX1BUQKmkSeref+AYLxmdlmtEsbF+Xd3cL2VCsmWgS6uHRiDgT4+aujEASLZodbOP56Y0Jeon77TxWCAJ3PjqGyhCiO645cT82zIa21TFmf27UwOz72sOJ7tyhCER+6FVEOhdAI1m6F7OffL5r9bN2nbxdVwVTLa6Nphr3mHExMtnQekMrKq2uwfv2y689MDBnfrnj6Tarl1upp+0XRRbhNA4vAEyXhyPV0R3Ejw9LnYxYnn+v39KnnkNTDIB4zC/65scTZB9mo3m101jfpama5exprwx1QELPlcFLiIyYBFyvPogY88o36QMsWXh1fYxZr/4lu/fkWSrHpqVQkAsJOWyReZpMKWenpBKgZtFRSDA2cAuBA4OoWscJ3RKokQAE9wZo+jcG7kt/HDLWp+UmLfKn+8Hpv6v48VE8TBPSep5qSROkM5kUOCDTu7jVSMae86HdIZ1L9d3yAyfPdOJKlMeqEeWiNb2JywwzZ+MwwY6pQFrs6eFp+s3yZmDlij+mAPbWAInF+7OTKa6f3pDRlK93c86g1Uzl1edHQerRK5WolUWvRW6aT1JZQE6Wrm6ou6kB20qLcuYQviYMXtCGGhPBEycj38jCNAZNNgKPqsOeCJddG5ARI+x2f2UH3QtOVFIgjUm7mREigikbwNR4Y0U5uZOQtJh3ZC1ZCz5IbXVcqyVGy6S8BNceoEJPTfU7YgmO158dT0='
)

# ======================
# S3 Configuration
# ======================
BUCKET_NAME = "cicd-validation-media"
PREFIX = "Valid_Files/"
THRESHOLD = 0.6
categories = ["educational", "entertainment", "sports", "tutorial", "news", "documentary"]
report_lines = []
has_failure = False

# Load models once
from whisper import load_model
whisper_model = load_model("base")

from transformers import CLIPProcessor, CLIPModel
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Start the report file
with open("validation_report.txt", "w") as report:
    report.write(" Validation Report\n")

    print("Fetching files from S3...")
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)

    if "Contents" not in response:
        print("No files found.")
    else:
        print(f"{len(response['Contents'])} files found.")

        for obj in response["Contents"]:
            key = obj["Key"]
            if key.endswith("/"):  # skip folders
                continue

            filename = os.path.basename(key)
            report.write(f" Processing: {filename}\n")

            try:
                s3_response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
                file_data = s3_response["Body"].read()
                mime_type, _ = mimetypes.guess_type(filename)

                if filename.endswith(('.mp4', '.mov', '.mkv', '.mp3', '.wav')):
                    with open("temp_media_file", "wb") as f:
                        f.write(file_data)
                    result = whisper_model.transcribe("temp_media_file")
                    transcript = result["text"]

                    result = classifier(transcript, candidate_labels=categories)
                    top_label = result['labels'][0]
                    top_score = result['scores'][0]

                elif filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    try:
                        image = Image.open(BytesIO(file_data))
                        image.verify()
                        image = Image.open(BytesIO(file_data))
                    except UnidentifiedImageError:
                        raise ValueError("Invalid image file.")

                    inputs = clip_processor(text=categories, images=image, return_tensors="pt", padding=True)
                    outputs = clip_model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)
                    top_index = probs.argmax().item()
                    top_label = categories[top_index]
                    top_score = probs[0][top_index].item()

                elif filename.endswith('.pdf'):
                    try:
                        reader = PdfReader(BytesIO(file_data))
                        text = ""
                        for page in reader.pages:
                            extracted = page.extract_text()
                            if extracted:
                                text += extracted
                        if not text.strip():
                            report.write(" No extractable text.\n\n")
                            continue
                        result = classifier(text, candidate_labels=categories)
                        top_label = result['labels'][0]
                        top_score = result['scores'][0]
                    except PdfReadError:
                        raise ValueError("Corrupted PDF.")

                else:
                    report.write(" Unsupported file type (skipped).\n\n")
                    continue

                report.write(f"Predicted Category: {top_label}\n")
                report.write(f"Confidence: {top_score * 100:.0f}%\n")

                if top_score < THRESHOLD:
                    report.write(f" Validation failed: Confidence Score is below ({THRESHOLD * 100:.0f}%)\n\n")
                    has_failure = True
                else:
                    report.write(" Validation passed.\n\n")

            except Exception as e:
                report.write(f" Error in processing file: {e}\n\n")
                has_failure = True

print("Validation completed. Report saved as validation_report.txt")

# Ensure GitHub fails the job if any file failed
if has_failure:
    import sys
    sys.exit(1)
