import gradio as gr
import requests
import json
import time
import os
from fastapi import FastAPI
from app.main import app as fastapi_app
from app.config import settings

# Initialize the FastAPI app
app = FastAPI()

# Mount the main FastAPI app
app.mount("/api", fastapi_app)

# Hugging Face Spaces configuration
HF_SPACE_URL = settings.HF_SPACE_URL

def process_invoices(files):
    try:
        # Upload files
        upload_url = f"{HF_SPACE_URL}/api/upload/"
        files_dict = [("files", (file.name, file.read(), file.type)) for file in files]
        response = requests.post(upload_url, files=files_dict)
        response.raise_for_status()
        
        task_id = response.json()["task_id"]
        
        # Poll for status
        status_url = f"{HF_SPACE_URL}/api/status/{task_id}"
        while True:
            status_response = requests.get(status_url)
            status_response.raise_for_status()
            
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
        
        csv_response.raise_for_status()
        excel_response.raise_for_status()
        
        # Save downloaded files
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"invoices_{task_id}.csv")
        excel_path = os.path.join(output_dir, f"invoices_{task_id}.xlsx")
        
        with open(csv_path, "wb") as f:
            f.write(csv_response.content)
        with open(excel_path, "wb") as f:
            f.write(excel_response.content)
        
        return f"Processing completed. Results saved as {csv_path} and {excel_path}"
    except requests.RequestException as e:
        return f"Error during processing: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# Define the Gradio interface
iface = gr.Interface(
    fn=process_invoices,
    inputs=gr.File(file_count="multiple", label="Upload Invoice Files (PDF, JPG, PNG, or ZIP)"),
    outputs="text",
    title=settings.PROJECT_NAME,
    description="Upload invoice files to extract and validate information. Results will be provided in CSV and Excel formats.",
)

# Combine FastAPI and Gradio
app = gr.mount_gradio_app(app, iface, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
