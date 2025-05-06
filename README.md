# Audio File Summarizer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://colab.research.google.com/github/renabracha/audio_file_summarizer/blob/main/Audio_file_summariser.ipynb)

## Acknowledgment
I would like to thank the following individuals and organisations that made this project possible.
* Groq for providing me free access to their API key and thereby allowing me to gain hands-on experience in making API calls without having to constantly worry about token limits.

## Audio Credits
Audio content from [Polyglot speaking in 7 languages](https://www.youtube.com/watch?v=esaXXVD0PTc), licensed under Creative Commons Attribution License (CC BY). Used for non-commercial, educational purposes.

## Abstract  
**Audio File Summarizer** generates structured summaries from multilingual audio recordings, making it ideal for extracting meeting minutes from business meetings. The program transcribes speech in multiple languages and outputs concise bullet-point summaries translated into a language of the user’s choice.  

For example, in a multilingual meeting between companies - where one party speaks Japanese, another Hebrew, and English serves as the common language - the program accurately identifies and transcribes all three, then produces summaries in the specified output language.  

### Workflow  
1. The user uploads an audio file to their My Drive on Google Drive.  
2. The program transcribes the audio, detecting all languages present.  
3. Key points are extracted from the transcriptions.  
4. Each point is summarized as a bullet item.  
5. The bullet list is translated into the user's chosen language.

## Development Notes  
* **Whisper** is used for transcription, processed in chunks to accurately capture short, multilingual exchanges.  
* Summarizing in the **original languages first**, then translating, improves efficiency and quality:  
  - Translation of raw transcripts is token-expensive and error-prone.  
  - Summarization reduces text size, speeding up translation and lowering costs.  
  - Preserving source-language context avoids misinterpretation due to premature translation.  
* **LangChain** provides modular sequencing for summarization and translation via structured prompt templates.  
Note: For greater privacy, local deployment with open-source models is recommended when working with sensitive recordings.

## Challenges  
* The most effective testing was done using a private, hour-long multilingual recording featuring Japanese, Hebrew, and English. However, for inclusion in this GitHub repository, it was necessary to find a publicly available audio file with a suitable license - an effort that proved time-consuming.
* Whisper’s **“large” model** performs best for multilingual detection. Smaller models struggle with non-English content.  
* **GPU acceleration** significantly boosts both speed and accuracy. On CPU, an 8-minute file takes about 15 minutes. On GPU (e.g., Google Colab’s T4), it processes in about half the time with better language detection.  
* Tuning **chunk size and overlap** was critical. Language switching in rapid, short exchanges meant chunk sizes had to be small (about 30 sec) with generous overlaps (about 10 sec). This increased the number of chunks but improved performance in mixed-language contexts.  
* **Streamlit** has limitations:  
  - Incompatibility with some `torch` versions (e.g., `2.6.0` CPU).  
  - Poor handling of multiple session states; more than two can cause flow issues, making it difficult to restart processes like uploading a new file.  
  - No functional "Start Over" button due to session state complexity.  
* Streamlit works better with **three or fewer variables**. Performance degrades with higher complexity.

<br>
The program is available in both Jupyter Notebook (.ipynb) and standalone Python script (.py) formats.

## Installation
To run audio_file_summarizer.py, do the following:

### Step 1. Place the files in a folder. 
1. Place the `.py` file in a local folder (e.g. `C:\temp\audio_file_summarizer`).
2. Create a file called `.env` and place the GROQ API key in the following format:
	`GROQ_API_KEY = <groq_api_key>`
3. Place the `.env` file in the same local folder. 

### Step 2. Install Python. 
1. In Windows, open the Command Prompt window.
2. Make sure Python is installed. In the Command Prompt window, type:
	`python --version`
If you get an error or "Python is not recognized", you need to install Python:
	1. Go to `https://www.python.org/downloads/`.
	2. Download the latest Python installer for Windows
	3. Run the installer and make sure to check `Add Python to PATH` during installation

### Step 3. Set up a virtual environment. 
This keeps your project dependencies isolated:
1. In the Command Prompt window, go to the script folder. Type:<br>
	`cd C:\<path to your script folder>`
2. In the Command Prompt, create a Python virtual environment named `audio_file_summarizer_env`.<br>
	`python -m venv audio_file_summarizer_env`
3. In the Command Prompt, activate the Python virtual environment.<br>
	`audio_file_summarizer_env\Scripts\activate`
4. Install the required dependencies.<br>
  `pip install -r requirements.txt`
5. Note that Whisper requires Ffmpeg to be installed. 

### Step 4. Run the script. 
1. In the Command Prompt window, run the Streamlit application. Type:<br>
	`streamlit run audio_file_summarizer.py`
<br>
<br>
This will start a local web server and open the application in your browser. Press the Analyse button to view the results. 

## Web app in action
### English
![Alt text for screen reader](https://github.com/renabracha/audio_file_summarizer/blob/main/screenshot_en1.jpg?raw=true)
![Alt text for screen reader](https://github.com/renabracha/audio_file_summarizer/blob/main/screenshot_en2.jpg?raw=true)
![Alt text for screen reader](https://github.com/renabracha/audio_file_summarizer/blob/main/screenshot_en3.jpg?raw=true)
![Alt text for screen reader](https://github.com/renabracha/audio_file_summarizer/blob/main/screenshot_en4.jpg?raw=true)

### Japanese
![Alt text for screen reader](https://github.com/renabracha/audio_file_summarizer/blob/main/screenshot_ja3.jpg?raw=true)
![Alt text for screen reader](https://github.com/renabracha/audio_file_summarizer/blob/main/screenshot_ja4.jpg?raw=true)