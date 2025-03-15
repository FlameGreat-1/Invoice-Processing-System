# Invoice Processing System

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [System Requirements](#system-requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [API Endpoints](#api-endpoints)
7. [Deployment](#deployment)
8. [Testing](#testing)
9. [Contributing](#contributing)
10. [License](#license)

## Introduction

The Invoice Processing System is a robust, industrial-grade solution designed to extract key details from vendor bills. It processes 80-100 invoices (1-2 pages each) in a single upload, utilizing advanced OCR and NLP techniques to accurately extract and validate invoice data.

## Features

- Supports PDF, JPG, PNG, and ZIP file uploads (max 100MB)
- Automatic multi-page invoice detection
- Extracts key fields: Invoice Number, Vendor Name & Address, Invoice Date, Totals
- Smart features:
  - Merges split pages using headers/totals
  - Validates amount calculations
  - Flags future dates and unapproved vendors
- Exports data to CSV and Excel formats
- Processes 100 pages in under 5 minutes
- Achieves 95%+ accuracy on clean scans
- Supports at least 3 different vendor formats

## System Requirements

- Python 3.9+
- Docker (for containerized deployment)
- 4GB RAM (minimum)
- 10GB free disk space

## Installation

1. Clone the repository:

2. Create a virtual environment:

3. Install dependencies:

4. Set up environment variables:

## Usage

1. Start the FastAPI server:

2. Open your browser and navigate to `http://localhost:8000/docs` to access the Swagger UI.

3. Use the `/upload/` endpoint to upload invoice files.

4. Check the processing status using the `/status/{task_id}` endpoint.

5. Download results using the `/download/{task_id}` endpoint.

## API Endpoints

- `POST /upload/`: Upload invoice files for processing
- `GET /status/{task_id}`: Check the status of a processing task
- `GET /download/{task_id}`: Download processed results (CSV or Excel)

For detailed API documentation, refer to the Swagger UI at `/docs` when the server is running.

## Deployment

### Using Docker

1. Build the Docker image:

2. Run the container:

### Deploying to Hugging Face Spaces

1. Create a new Space on Hugging Face.
2. Connect your GitHub repository to the Space.
3. Configure the Space to use the Dockerfile for deployment.
4. Push your changes to GitHub, and Hugging Face will automatically deploy your app.

## Testing

Run the test suite:


## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
