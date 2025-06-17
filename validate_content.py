import os
import mimetypes
from transformers import pipeline
from PIL import Image, UnidentifiedImageError
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

THRESHOLD = 0.9
folder_path = "Media_Files"
categories = ["educational", "entertainment", "sports", "tutorial", "news", "documentary"]

report_lines = []
has_failure = False  #  Flag to decide final build status

# Start the report file
with open("validation_report.txt", "w") as report:
    report.write(" Validation Report\n")
    

    for file_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file_name)
        file_type, _ = mimetypes.guess_type(full_path)

        report.write(f" Processing: {file_name}\n")

        try:
            if file_name.endswith(('.mp4', '.mov', '.mkv', '.mp3', '.wav')):
                from whisper import load_model
                model = load_model("base")
                result = model.transcribe(full_path)
                transcript = result["text"]

                classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
                result = classifier(transcript, candidate_labels=categories)
                top_label = result['labels'][0]
                top_score = result['scores'][0]

            elif file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image = Image.open(full_path)
                image.verify()
                image = Image.open(full_path)

                from transformers import CLIPProcessor, CLIPModel
                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

                inputs = processor(text=categories, images=image, return_tensors="pt", padding=True)
                outputs = model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
                top_index = probs.argmax().item()
                top_label = categories[top_index]
                top_score = probs[0][top_index].item()

            elif file_name.endswith('.pdf'):
                reader = PdfReader(full_path)
                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted
                if not text.strip():
                    report.write(" No extractable text.\n\n")
                    continue
                classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
                result = classifier(text, candidate_labels=categories)
                top_label = result['labels'][0]
                top_score = result['scores'][0]

            else:
                report.write(" Unsupported file type.\n\n")
                #continue
                has_failure = True

            report.write(f"Predicted Category: {top_label}\n")
            report.write(f"Confidence: {top_score * 100:.0f}%\n")

            if top_score < THRESHOLD:
                report.write(f" Validation failed: Confidence Score is below  threshold ({THRESHOLD * 100:.0f})\n\n")
                has_failure = True
            else:
                report.write(" Validation passed.\n\n")

        except Exception as e:
            report.write(f" Error in processing file: {e}\n\n")
            has_failure = True

print(" Validation completed. Report saved as validation_report.txt")

# Ensure GitHub gets to upload the file before exiting
if has_failure:
    import sys
    sys.exit(1)

