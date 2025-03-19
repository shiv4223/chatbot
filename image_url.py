import os
import uuid
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pydrive2.auth import ServiceAccountCredentials

def upload_file_to_google_drive(image_file):
    """
    Uploads an image file to Google Drive using a Service Account.
    
    Parameters:
        image_file (FileStorage or str): FileStorage object from Flask or file path.

    Returns:
        str: Public Google Drive file URL.
    """
    # Handle Flask FileStorage object
    if hasattr(image_file, "save"):
        temp_path = f"/tmp/{uuid.uuid4()}{os.path.splitext(image_file.filename)[-1]}"
        
        # Ensure the file pointer is at the beginning before saving
        image_file.seek(0)
        image_file.save(temp_path)
        image_path = temp_path
    elif isinstance(image_file, str):
        image_path = image_file
        if not os.path.exists(image_path):
            raise FileNotFoundError("Image file not found!")
    else:
        raise TypeError("Invalid file type. Must be FileStorage or file path string.")
    
    # Ensure the file is fully saved before upload
    if os.path.getsize(image_path) == 0:
        raise ValueError("Error: The saved file is empty!")
    
    # Generate a unique filename with the correct extension
    filename = f"{uuid.uuid4()}{os.path.splitext(image_path)[-1]}"
    
    # Authenticate with Google Drive using a Service Account
    creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scopes=["https://www.googleapis.com/auth/drive"])
    gauth = GoogleAuth()
    gauth.credentials = creds
    drive = GoogleDrive(gauth)

    # Upload file to the shared folder
    file = drive.CreateFile({"title": filename, "parents": [{"id": "1Z0uP7DbN-WfUxNC2YWAXNPhVxn_Kk2vY"}]})  # Replace YOUR_FOLDER_ID
    
    # Ensure file is readable before uploading
    with open(image_path, "rb") as f:
        file.SetContentFile(image_path)
        file.Upload()
    
    # Make file public
    file.InsertPermission({
        'type': 'anyone',
        'value': 'anyone',
        'role': 'reader'
    })
    
    # Remove temporary file if it was created
    if hasattr(image_file, "save"):
        os.remove(temp_path)

    return f"https://drive.google.com/uc?id={file['id']}"

