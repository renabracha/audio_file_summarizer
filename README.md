# Audio File Summarizer

## Acknowledgment
I would like to thank the following individuals and organisations that made this project possible.
* Groq for providing me free access to their API key and thereby allowing me to gain hands-on experience in making API calls without having to constantly worry about token limits.

## Audio Credits
Audio content from [Polyglot speaking in 7 languages](https://www.youtube.com/watch?v=esaXXVD0PTc), licensed under Creative Commons Attribution License (CC BY). Used for non-commercial, educational purposes.

## Abstract
Audio file summarizer produces a list of summaries from an audio file in the language of the user's choice. It is useful for extracting meeting minutes out of the recording of a business meeting. The program can handle audio files containing multiple languages. In a business meeting between two companies with one party speaking in Japanese, another in Hebrew, and the common language being English, the program transcribes the utterances in all three languages, and produces meeting minutes in the language specified by the user.
<br>
The program works in the following sequence:
1. Ask the user to upload an audio file to their My Drive on Google Drive.
2. Transcribe the video in all the languages heard in the audio file.
3. Extract the key points.
4. Summarise each point in a bullet list.
5. Translate the bullet list into the language of the user's choice.

## Points of Consideration
We summarize in the original languages first, then translate the summary since it is generally more efficient and makes good practical sense.
* Translation has higher token-level cost (especially for long texts) in both time and API usage.
* Summarization reduces content size, which speeds up and simplifies the translation task.
* Summarizing in the original languages also preserves contextual and cultural nuances, which often get muddled if you translate first.

## Challenges
* It took a long time to find a suitable multilingual audio file with an appropriate license that allows me to use in this project. An hour-long audio recording of a business meeting involving Japanese, Hebrew and English produced the best result. The model did not detect all the languages successfully in a couple of YouTube videos, including the one shown in the result.
* The "large" sized Whisper model processes audio files the best. The "medium", "small" and "base" sized models all show difficulty at detecting non-English languages.
* Running Whisper without a GPU affects its performance greatly, not just in terms of speed but also in terms of language detection performance. It takes about 15 minutes to process an 8-minute long audio file on the CPU. It takes about half that time to process the same file, and manages to show a better language detection rate when run on the GPU. If your PC is not equipped with up-to-date GPU card that is compatible with the GPU-enabled version of PyTorch, it is preferable to run the program in Google Colab using T4. The .ipynb file is provided in the repo folder (no Streamlit section).  
* I had to experiment iteractively and adjust the chunk size in milliseconds until I found an optimal size for capturing the conversational segments of various length. If a converstaional exchange in one language is short and that exchange goes into a chunk, together with part of the subsequent conversational exchange in a different language, Whisper failed to identify the language of the shorter segment. It meant each chunk was small (30 seconds), the overlap felt rather large (10 seconds) and the number of chunks being produced ended up being high, but it succeeded in coping with the frequent switchings in languages.
* Streamlit seems to be incompatibile with certain versions of torch (e.g. 2.6.0 CPU version). Upgrading the torch installation did not help. 
* Streamlit also seems to find it difficult to cope with seven states. Up to two states is fine, but if there are more (I have not tested its threshold when it stops working correctly), it cannot return to the beginning of the loop to process another audio file where the starting state is clearly labelled and specified. Claude also could not find a workaround, hence the missing Start Over button in the program. Streamlit can work correctly with three variables. 


# Installation
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
![Alt text for screen reader](https://github.com/renabracha/audio_file_summarizer/blob/main/screenshot_ja1.jpg?raw=true)
![Alt text for screen reader](https://github.com/renabracha/audio_file_summarizer/blob/main/screenshot_ja2.jpg?raw=true)
![Alt text for screen reader](https://github.com/renabracha/audio_file_summarizer/blob/main/screenshot_ja3.jpg?raw=true)
![Alt text for screen reader](https://github.com/renabracha/audio_file_summarizer/blob/main/screenshot_ja4.jpg?raw=true)