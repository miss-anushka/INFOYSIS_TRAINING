from flask import Flask, request, jsonify, render_template, send_file
import os
from ultralytics import YOLO
from easyocr import Reader
import zipfile
import pandas as pd
import cv2
from fuzzywuzzy import fuzz, process
import re
import numpy as np
import sqlite3

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULTS_FOLDER'] = 'results/'

classifier = YOLO("/Users/anushkamishra/runs/classify/train7/weights/best.pt")
detector = YOLO("/Users/anushkamishra/runs/detect/train3/weights/best.pt")
reader = Reader(['en'])

# Initialize the database
def init_db():
    with sqlite3.connect("results.db") as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                address TEXT,
                uid TEXT,
                is_aadhaar BOOLEAN
            )
        ''')
        conn.commit()

# Insert data into the database
def insert_into_db(name, address, uid, is_aadhaar):
    with sqlite3.connect("results.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO results (name, address, uid, is_aadhaar) VALUES (?, ?, ?, ?)",
            (name, address, uid, is_aadhaar),
        )
        conn.commit()

# Process uploaded images
def process_image(image_path):
    if classifier.predict(image_path)[0].probs.numpy().top1 == 0:
        fields = detector(image_path)
        image = cv2.imread(image_path)
        extracted_data = {}
        for field in fields[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = map(int, field[:6])
            field_class = detector.names[class_id]
            cropped_roi = image[y1:y2, x1:x2]
            gray_roi = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)
            text = reader.readtext(gray_roi, detail=0)
            extracted_data[field_class] = ' '.join(text)
        return extracted_data
    return None

# Normalize text for matching
def normalize_text(text):
    if not text:
        return "text empty"
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split()).lower()

# Calculate match score using fuzzy logic
def calculate_match_score(input_value, extracted_value):
    if pd.isna(input_value) or pd.isna(extracted_value):
        return 0
    return fuzz.ratio(str(input_value), str(extracted_value))

# Match name logic
def name_match(input_name, extracted_name):
    if extracted_name is None:
        return False
    input_name = normalize_text(input_name)
    extracted_name = normalize_text(extracted_name)

    if input_name == extracted_name:
        return True

    input_parts = input_name.split()
    extracted_parts = extracted_name.split()

    if sorted(input_parts) == sorted(extracted_parts):
        return True

    if len(input_parts) == 2 and len(extracted_parts) == 3:
        if input_parts[0] == extracted_parts[0] and input_parts[1] == extracted_parts[2]:
            return True
    if len(input_parts) == 3 and len(extracted_parts) == 2:
        if extracted_parts[0] == input_parts[0] and extracted_parts[1] == input_parts[2]:
            return True

    for part in input_parts:
        if part not in extracted_parts:
            return False
    return True

# Address matching logic
def address_match(input_address, extracted_address):
    if input_address is None or extracted_address is None:
        return False, 0.0, {}

    if isinstance(input_address, pd.Series):
        input_address = input_address.to_dict()

    extracted_address = normalize_text(extracted_address)
    final_score = 0
    weights = {
        "State": 0.2,
        "Landmark": 0.2,
        "Premise Building Name": 0.2,
        "City": 0.2,
        "Street Road Name": 0.1,
        "Floor Number": 0.05,
        "House Flat Number": 0.05
    }
    tokens = extracted_address.split(" ")
    for field, weight in weights.items():
        input_value = input_address.get(field, "")
        match_score = fuzz.token_set_ratio(normalize_text(input_value), extracted_address) if input_value else 0
        input_address[field + " Match Score"] = match_score
        final_score += match_score * weight
    pincode_score = process.extractOne(input_address.get("PINCODE"), tokens)[1]
    input_address['PINCODE Match Score'] = pincode_score
    pincode_matched = True if input_address['PINCODE Match Score'] == 100 else False

    return final_score >= 70 and pincode_matched, final_score, input_address

# Compare data between input and extracted values
def compare_data(input_data, json_data):
    excel_data = input_data.copy()
    excel_data['Accepted/Rejected'] = ""
    excel_data['Document Type'] = ""
    excel_data['Final Remarks'] = ""

    for idx, row in excel_data.iterrows():
        serial_no = row.get("SrNo")
        uid = row.get("UID")
        extracted = json_data.get(serial_no)

        if extracted:
            extracted_uid = extracted.get("uid", "").replace(" ", "")
            extracted_name = extracted.get("name", "")
            extracted_address = extracted.get("address", "")
            is_aadhaar = uid == extracted_uid and name_match(row.get("Name"), extracted_name) and address_match(row, extracted_address)[0]

            insert_into_db(extracted_name, extracted_address, extracted_uid, is_aadhaar)

            row['UID Match Score'] = 100 if uid == extracted_uid else 0
            row['Name Match Score'] = calculate_match_score(row.get("Name"), extracted_name)
            address_match_result, address_score, _ = address_match(row, extracted_address)
            row['Final Remarks'] = "All matched" if is_aadhaar else "Mismatch"
            row['Document Type'] = "Aadhaar" if is_aadhaar else "Non-Aadhaar"
            row['Accepted/Rejected'] = "Accepted" if is_aadhaar else "Rejected"

        else:
            row['Final Remarks'] = "Non Aadhar"
            row['Document Type'] = "Non Aadhar"
            row['Accepted/Rejected'] = "Rejected"

        excel_data.loc[idx] = row
    return excel_data

@app.route('/')
def home():
    return render_template('front_page.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'zipfile' in request.files and 'excelfile' in request.files:
        zip_file = request.files['zipfile']
        excel_file = request.files['excelfile']

        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], zip_file.filename)
        excel_path = os.path.join(app.config['UPLOAD_FOLDER'], excel_file.filename)
        zip_file.save(zip_path)
        excel_file.save(excel_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(app.config['UPLOAD_FOLDER'])

        image_paths = [os.path.join(app.config['UPLOAD_FOLDER'], f) for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith(('.jpg', '.png'))]
        processed_results = {}

        for image_path in image_paths:
            file_name = os.path.basename(image_path)
            key = file_name.split('.')[0][:3]
            extracted_data = process_image(image_path)

            if extracted_data:
                if key in processed_results:
                    processed_results[key] = {**processed_results[key], **extracted_data}
                else:
                    processed_results[key] = extracted_data

        df = pd.read_excel(excel_path)
        df = df.astype('str')
        comparison_results = compare_data(df, processed_results)

        results_to_display = comparison_results[['SrNo', 'Accepted/Rejected', 'Document Type', 'Final Remarks']]
        results_file_path = os.path.join(app.config['RESULTS_FOLDER'], 'results.xlsx')
        os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
        results_to_display.to_excel(results_file_path, index=False)

        return jsonify({"message": "Files processed successfully!", "results": results_to_display.to_dict(orient='records')})

    return jsonify({"error": "Both files are required."}), 400

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
