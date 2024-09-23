# Computer vision CLIP Service for COmputer Vision Weekend project Group 16

This file uses the CLIP service to determine if an animal image is dangerous or not.

This file is the backend companion of https://github.com/Encode-Group16-AI-bootcamp-Q3-2024/computer-vision-weekend-project

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- virtualenv (recommended for creating isolated Python environments)

## Setup and Installation

Follow these steps to set up the project environment and install the necessary packages:

1. Clone the repository:
   ```
   git clone git@github.com:Encode-Group16-AI-bootcamp-Q3-2024/computer-vision-clipserver.git
   cd computer-vision-clipserver
   ```

2. Create a virtual environment:
   ```
   python -m venv myenv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     myenv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source myenv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install torch torchvision clip flask pillow
   ```

## Running the Code

Run the server before run the Next JS Project.

```
python analyze-animal.py
```

The server will expose the following URL: http://localhost:5000/analyze Use it in the project env CLIP_API_URL
```
CLIP_API_URL=http://localhost:5000/analyze
```

