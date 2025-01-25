UID Aadhar Fraud Detection System
This project is a system designed to detect and process fraudulent Aadhar card submissions. It uses Flask for the backend and HTML/CSS for the frontend. Below is an overview of the project.

Features
Image Classification: Determines if the uploaded image is an Aadhar card. Non-Aadhar images are rejected.
OCR Processing: Extracts text from valid Aadhar images.
Input: Accepts an Excel file and a ZIP file with images for processing.
Fraud Detection: Compares extracted text with the provided Excel data and generates matching scores.
Results:
Displays a table showing UID, matching scores, and other details.
Allows users to download a detailed Excel report.
Workflow
Upload an Excel file and ZIP file: The system processes the input files.
Image Classification: Identifies Aadhar images and rejects non-Aadhar images.
Text Extraction: OCR extracts details from valid Aadhar images.
Data Matching: Compares extracted data with the reference Excel file.
Results:
A table with matching scores and UIDs is displayed on the frontend.
Users can download the results in an Excel format.
Installation
Clone the repository and navigate to the project folder.
Set up a Python virtual environment and install the dependencies listed in requirements.txt.
Run the Flask app and access the application in your browser.
Technologies Used
Backend: Flask
Frontend: HTML, CSS
OCR: Tesseract OCR or equivalent
Data Processing: Pandas, OpenPyXL
Outputs
A table displaying UIDs, matching scores, and statuses.
A downloadable Excel file with detailed results.
