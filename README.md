# Erica: The AI Tutor

### How to Run Locally
Prerequisites

* Node.js and npm installed.
* Python installed.
* An OpenAI API Key.

## Backend Setup

Open a terminal and navigate to the backend/ directory.

Create the environment file: Create a file named .env in the backend/ folder and paste your OpenAI key inside:

```Plaintext
OPENAI_API_KEY=your_key_here
```

### Set up the Python environment: Create a virtual environment and install the required dependencies.
```Bash

# Create virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# Create virtual environment (Mac/Linux)
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies:
```bash

pip install -r requirements.txt
```

### Run the server:
```Bash
python userinput.py
```

The backend should now be running on http://localhost:5000.

## Frontend Setup

Open a new terminal window (keep the backend terminal running) and navigate to the project root (where package.json is located).

Install Node packages:

```Bash

npm install
```
### Run the application: Since this is a Vite app, use the dev command:
```bash

npm run dev
```
Open in Browser: Click the link shown in the terminal (usually http://localhost:5173) to view the app.