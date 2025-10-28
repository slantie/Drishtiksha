# Drishtiksha CLI

## 1. Overview

The **Drishtiksha CLI** is a powerful, professional command-line interface designed for developers, researchers, and power users to interact directly with the Drishtiksha ML inference engine. It provides a fast, scriptable, and detailed way to analyze media files without needing the web interface.

**Key Features:**

*   **Direct Access:** Run analysis directly from your terminal.
*   **Lazy Loading:** Starts instantly and only loads the required AI models on demand.
*   **Multiple Analysis Modes:** Run a full ensemble of models, a single specific model, or interactively select a custom set.
*   **Batch Processing:** Analyze entire directories of media files with a single command.
*   **Rich Output:** View results in beautifully formatted tables or export them as structured JSON for further processing.
*   **System Inspection:** Check system health, GPU status, and view details of all available models.

## 2. Installation & Setup

The CLI is automatically installed as part of the `Server` module's Python environment.

1.  Navigate to the `Server/` directory.
2.  Activate the virtual environment: `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows).
3.  The CLI is now available via the `drishtiksha` command.

You can verify the installation by running:
```bash
drishtiksha --version
```

## 3. Core Concepts

### 3.1. Lazy Loading

The CLI is optimized for speed. When you run a command, it **does not** load all AI models into memory. It only loads the specific model(s) required for your requested analysis, which makes it fast to start and memory-efficient.

### 3.2. Model Selection

You have three primary ways to select which models to use for an analysis:

| Mode | Command | Behavior | Use Case |
| :--- | :--- | :--- | :--- |
| **Ensemble (Default)** | `drishtiksha analyze <file>` | Automatically uses **all** available models compatible with the media type. | Best for a robust, final verdict based on a consensus of models. |
| **Single Model** | `drishtiksha analyze <file> --model <model_name>` | Uses only the one specified model. | Fast, targeted analysis; useful for testing a specific model. |
| **Interactive** | `drishtiksha analyze <file> --custom` or `drishtiksha interactive` | Prompts you with a checklist to select one or more models. | Perfect for comparing the performance of a specific subset of models. |

### 3.3. Output Formats

The CLI provides two main output formats:

1.  **Console Tables (Default):** Results are printed in clean, human-readable tables using the `rich` library.
2.  **JSON (`--output` flag):** Results are saved to a file in a structured JSON format, ideal for scripting, data analysis, or integration with other tools.

---

## 4. Command Reference

### **`drishtiksha analyze`**

Analyzes a single media file. This is the primary command you will use.

**Syntax:** `drishtiksha analyze <file_path> [options]`

**Arguments:**

*   `<file_path>` (Required): The path to the video, audio, or image file you want to analyze.

**Options:**

*   `-m, --model <model_name>`: Specify a single model to use (e.g., `AUDIO-MFT-CRNN-V1`).
*   `-c, --custom`: Enter interactive mode to select multiple models from a list.
*   `-v, --visualize`: Generate and save a visualization artifact (e.g., an overlay video or a noise map).
*   `-o, --output <path>`: Save the detailed analysis results to a JSON file.
*   `--verbose`: Display more detailed metrics and logs in the console.

**Examples:**

*   **Analyze a video with all compatible models (Ensemble):**
    ```bash
    drishtiksha analyze /path/to/my_video.mp4
    ```

*   **Analyze an audio file with our newly integrated model:**
    ```bash
    drishtiksha analyze /path/to/my_audio.wav --model AUDIO-MFT-CRNN-V1
    ```

*   **Analyze an image and generate a visualization (e.g., DIRE map):**
    ```bash
    drishtiksha analyze /path/to/an_image.png --model DISTIL-DIRE-V1 --visualize
    ```

*   **Compare two specific models on a video and save the result:**
    ```bash
    drishtiksha analyze /path/to/test.mov --custom --output comparison.json
    # (This will prompt you to select models from a checklist)
    ```

### **`drishtiksha batch`**

Analyzes all media files within a directory.

**Syntax:** `drishtiksha batch <directory_path> [options]`

**Arguments:**

*   `<directory_path>` (Required): The path to the folder containing media files.

**Options:**

*   `-r, --recursive`: Search for media files in all subdirectories as well.
*   `-m, --model <model_name>`: Use a single, specific model for all files. If omitted, the best model for each file's type is auto-selected.
*   `-o, --output <path>`: Save the aggregated results for all files into a single JSON file.
*   `-p, --parallel <count>`: Number of parallel analysis jobs to run (default: 1). *Note: Requires significant CPU/GPU resources.*

**Examples:**

*   **Analyze all videos and audio files in a directory:**
    ```bash
    drishtiksha batch ./my_dataset/
    ```

*   **Analyze all images recursively and save a report:**
    ```bash
    drishtiksha batch ./all_my_images/ --recursive --model MFF-MOE-V1 --output image_report.json
    ```

### **`drishtiksha models`**

Lists all available AI models that are configured in the system.

**Syntax:** `drishtiksha models [options]`

**Options:**

*   `-t, --type <type>`: Filter the list by media type. Options: `video`, `audio`, `image`.
*   `-v, --verbose`: Show detailed information for each model, including its Python class and file paths.

**Examples:**

*   **List all available models:**
    ```bash
    drishtiksha models
    ```

*   **List only the audio models:**
    ```bash
    drishtiksha models --type audio
    ```

### **`drishtiksha stats`**

Displays the current system health and resource usage.

**Syntax:** `drishtiksha stats [options]`

**Options:**

*   `-d, --detailed`: Shows more detailed system information.

**Example:**

*   **Check system status, including CPU, RAM, and GPU memory usage:**
    ```bash
    drishtiksha stats
    ```

### **`drishtiksha interactive`**

Starts a guided, step-by-step wizard for analyzing a single file. Perfect for first-time users.

**Syntax:** `drishtiksha interactive`

---

## 5. Practical Workflows

Here’s how to use the CLI to accomplish common tasks.

#### **Workflow 1: Quick Sanity Check on a Single Video**

**Goal:** Quickly get a robust verdict on a single video file.

**Command:** Use the default ensemble mode.
```bash
drishtiksha analyze "path/to/suspicious_video.mp4"
```
**Result:** The CLI will run all compatible video models and provide a final "Ensemble Prediction" based on a majority vote, giving you the most reliable result.

#### **Workflow 2: In-Depth Forensic Analysis**

**Goal:** Deeply investigate a video and get a detailed report with visualizations.

**Command:** Use the `analyze` command with the `--visualize` and `--output` flags.
```bash
drishtiksha analyze "critical_video.mov" --visualize --output "forensic_report.json"
```
**Result:** This performs a full ensemble analysis, saves a detailed JSON report with per-frame data, and creates a new video file with a real-time analysis graph overlaid on it.

#### **Workflow 3: Processing an Entire Dataset**

**Goal:** Analyze a large folder of audio files from a research dataset and compile the results.

**Command:** Use the `batch` command with `--recursive` and `--output`.
```bash
drishtiksha batch ./asv_spoof_dataset/ --recursive --output asv_results.json
```
**Result:** The CLI will find every audio file in the dataset, analyze each one using the best available audio model, and save all predictions into a single, structured `asv_results.json` file for easy analysis.