import sys
import os
import uvicorn

# Get the absolute path of the project root
project_root = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, project_root)

os.environ["PYTHONPATH"] = project_root

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)