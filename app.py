import gradio as gr
import requests
import json
import time
from fastapi import FastAPI
from app.main import app as fastapi_app

# Initialize the FastAPI app
app = FastAPI()

# Mount the main FastAPI app
app.mount("/api", fastapi_app)

# Hugging Face Spaces configuration
HF_SPACE_URL = "https://huggingface.co/spaces/your-username/your-space-name"

def process_invoices(files):
    # Upload files
    upload_url = f"{HF_SPACE_URL}/api/upload/"
    files_dict = [("files", (file.name, file.read(), file.type)) for file in files]
    response = requests.post(upload_url, files=files_dict)
    if response.status_code != 200:
        return f"Error uploading files: {response.text}"
    
    task_id = response.json()["task_id"]
    
    # Poll for status
    status_url = f"{HF_SPACE_URL}/api/status/{task_id}"
    while True:
        status_response = requests.get(status_url)
        if status_response.status_code != 200:
            return f"Error checking status: {status_response.text}"
        
        status_data = status_response.json()
        if status_data["status"]["status"] == "Completed":
            break
        elif status_data["status"]["status"] == "Failed":
            return f"Processing failed: {status_data['status']['message']}"
        
        time.sleep(5)  # Wait 5 seconds before checking again
    
    # Download results
    csv_url = f"{HF_SPACE_URL}/api/download/{task_id}?format=csv"
    excel_url = f"{HF_SPACE_URL}/api/download/{task_id}?format=excel"
    
    csv_response = requests.get(csv_url)
    excel_response = requests.get(excel_url)
    
    if csv_response.status_code != 200 or excel_response.status_code != 200:
        return "Error downloading results"
    
    # Save downloaded files
    csv_path = f"invoices_{task_id}.csv"
    excel_path = f"invoices_{task_id}.xlsx"
    
    with open(csv_path, "wb") as f:
        f.write(csv_response.content)
    with open(excel_path, "wb") as f:
        f.write(excel_response.content)
    
    return f"Processing completed. Results saved as {csv_path} and {excel_path}"

# Define the Gradio interface
iface = gr.Interface(
    fn=process_invoices,
    inputs=gr.File(file_count="multiple", label="Upload Invoice Files (PDF, JPG, PNG, or ZIP)"),
    outputs="text",
    title="Invoice Processing System",
    description="Upload invoice files to extract and validate information. Results will be provided in CSV and Excel formats.",
)

# Combine FastAPI and Gradio
app = gr.mount_gradio_app(app, iface, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
    