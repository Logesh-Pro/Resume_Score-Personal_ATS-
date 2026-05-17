import os
from pypdf import PdfReader
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(
    __name__,
    template_folder="../templates"
)

app.config["UPLOAD_FOLDER"] = "/tmp"

def extract_text_from_pdf(pdf_path):
    text = ""

    reader = PdfReader(pdf_path)

    for page in reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted

    return text

def compute_similarity(resume_text, job_desc):

    documents = [resume_text, job_desc]

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )

    return round(similarity[0][0] * 100, 2)

@app.route("/", methods=["GET", "POST"])
def index():

    score = None

    if request.method == "POST":

        if "resume" not in request.files:
            return "No file uploaded", 400

        file = request.files["resume"]

        job_desc = request.form["job_desc"]

        if file.filename == "" or job_desc.strip() == "":
            return "Invalid input", 400

        filename = secure_filename(file.filename)

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        resume_text = extract_text_from_pdf(filepath)

        score = compute_similarity(
            resume_text,
            job_desc
        )

    return render_template(
        "index.html",
        score=score
    )

handler = app

if __name__ == "__main__":
    app.run(debug=True)