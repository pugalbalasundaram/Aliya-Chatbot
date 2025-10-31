import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORM_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'forms'))


print("Form Folder Path:", FORM_FOLDER)
print("Files in FORM_FOLDER:", os.listdir(FORM_FOLDER))
