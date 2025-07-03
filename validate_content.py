import os
import mimetypes
import subprocess
from io import BytesIO
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, pipeline
import whisper
import boto3

# Load models
whisper_model = whisper.load_model("base")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# AWS S3 client
s3 = boto3.client("s3",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    aws_session_token=os.environ["AWS_SESSION_TOKEN"]
)

# Categories and config
categories = ["educational", "entertainment", "sports", "news", "documentary"]
BUCKET = "cicd-validation-media"
PREFIX = "Valid_Files/"
THRESHOLD = 0.8
has_failure = False

def extract_frame(video_path, frame_path="frame.jpg"):
    subprocess.run([
        "ffmpeg", "-i", video_path, "-ss", "00:00:01.000", "-vframes", "1", frame_path,
        "-y", "-loglevel", "quiet"
    ])

def classify_frame(image_path):
    image = Image.open(image_path)
    inputs = clip_processor(text=categories, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    top_index = probs.argmax().item()
    return categories[top_index], probs[0][top_index].item()

# Begin validation
response = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)
for obj in response.get("Contents", []):
    key = obj["Key"]
    if key.endswith("/"):
        continue

    print(f"🔍 Processing: {key}")
    file_obj = s3.get_object(Bucket=BUCKET, Key=key)
    file_data = file_obj["Body"].read()
    filename = os.path.basename(key)
    mime_type, _ = mimetypes.guess_type(filename)

    with open("temp_media", "wb") as f:
        f.write(file_data)

    try:
        if filename.endswith(('.mp4', '.mov', '.mkv', '.mp3', '.wav')):
            result = whisper_model.transcribe("temp_media")
            transcript = result["text"].strip()

            # Use Whisper if valid text exists
            if transcript and len(transcript.split()) > 5:
                res = classifier(transcript, candidate_labels=categories)
                top_label = res["labels"][0]
                top_score = res["scores"][0]
            else:
                print("⚠️ No clear audio — using CLIP fallback")
                extract_frame("temp_media")
                top_label, top_score = classify_frame("frame.jpg")

        elif filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            image = Image.open(BytesIO(file_data))
            inputs = clip_processor(text=categories, images=image, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            top_index = probs.argmax().item()
            top_label = categories[top_index]
            top_score = probs[0][top_index].item()

        elif filename.endswith('.pdf'):
            # PDF handling (same as your current logic)
            pass

        else:
            print(f"❌ Unsupported file type: {filename}")
            continue

        print(f"✅ Category: {top_label} ({top_score * 100:.0f}%)")

        if top_score < THRESHOLD:
            print(f"❌ Validation failed: score below {THRESHOLD * 100:.0f}%")
            has_failure = True

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        has_failure = True

# Exit with failure if needed
if has_failure:
    import sys
    sys.exit(1)
