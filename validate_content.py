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
    aws_access_key_id='ASIASRGASLYJVJTIFAEK',
    aws_secret_access_key='9lnWMadJe4zoZo5t24MqahbbaLV+y5bz+ALoOKQc',
    aws_session_token='IQoJb3JpZ2luX2VjEM3//////////wEaCmFwLXNvdXRoLTEiRzBFAiAFUEki4ADIdFJzXIBSwCV+DYn5amURqIONZU0G7e8pbgIhALzOty7njyx9xU/qP3XmeWFsHkmgfO8guI6GtmWhkI46KrcDCMb//////////wEQABoMMTc0MzUwMDMyNDAzIgxv3oFicjGBexMuhG4qiwM9//2lbtmghjuxsi1S8oaNIeLCKS2xV9mKav6ALbl6YT912B84h/k8fja5GD382K66tQfEVoTLQLd6KlXmmNKbFTWC0PZdE+FT3n7jK7TGHJJP0JayqLUglp5LQxMK7n7v6IlwxVxZBa8wEnqTTX7a/K/TB6jJztlQSD+9/MoA20Wy7lIOKQzvMEvmNKMtsrEqid76QdoZLfqlXp9BOOS66wccuRS1scTrDo8M8w0kglJpuG7RZwQUrP1ySrj0xdLY/BuGG39EAyX8g6PRDnm6ilK0F2SnXvISG4gihx7CW0gBUyVkGUkKSHPlVghskxoFj5PyecEpe8+lp2wmdMKRCw69ZxhyukzoqhQqDYlKW+DJC1+WnC/6KYH+auQOnnSObLT8qyyvrLkk8w6/KpLjOrK6GIAA8koTtQWvqAg8ZdcxjBSVe6IANHx6VsyNG/0lf+QGb8tCW/FCW2Tc0WTJjyaSqGBfj71nRo8adYfW/t0SbtTIo+t2TybRxKuv2+KikQTPA7dTl02q6TCU/IvDBjqmAe9i569yHMhBtRsRtzYAWm+FkEOn8apmivhBcXop+Gh/YcDRB1Ca3vr7sofuVr4U/aKUh9WStW6nJSufMkJYMO1HHzKFvotJ3KB6VSWjtOotElUHdc9rUgcCE6EBQIlSlkCuARxlQZNTmaKb5lFj/qcfv4T0KwCI2J3wB92NdwiyGGpqL2Qm9XDg3e1gK2h0FqLRoBozlKZ71EsaAKPVaYpgxsW43aE='
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
