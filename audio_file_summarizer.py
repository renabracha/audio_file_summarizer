import os
import re
import torch
import whisper
import numpy as np
import logging
from tqdm import tqdm
from pydub import AudioSegment
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
import contextlib
import io
import datetime
import sys

# Set up the environment variables
load_dotenv()

# Initialize the language model
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Make a transcript of an audio file in chunks of 5 minutes
# Introduce a 20-second overlap between two adjacent chunks to maintain context continuity
import pydub

def make_transcript_in_chunks(audio_file_path, chunk_duration_ms=30000, overlap_ms=10000):
    """
    Transcribe an audio file by splitting it into overlapping chunks to avoid memory issues.

    Args:
        audio_file_path: Path to the audio file
        chunk_duration_ms: Duration of each chunk in milliseconds (30 seconds)
        overlap_ms: Overlap between chunks in milliseconds (10 seconds)

    Returns:
        Full transcript of the audio file
    """
    print(f"Loading audio file: {audio_file_path}")

    audio = AudioSegment.from_file(audio_file_path)

    # Get audio duration in milliseconds
    audio_duration = len(audio)
    print(f"Audio duration: {audio_duration/1000:.2f} seconds")

    # Create a temporary directory for chunks if it doesn't exist
    if not os.path.exists("temp_chunks"):
        os.makedirs("temp_chunks")

    # Calculate positions for chunking (with overlap)
    chunk_positions = list(range(0, audio_duration, chunk_duration_ms - overlap_ms))

    # Ensure last chunk doesn't exceed audio length
    if chunk_positions and chunk_positions[-1] + chunk_duration_ms > audio_duration:
        chunk_positions[-1] = max(0, audio_duration - chunk_duration_ms)

    # Add the final chunk if needed
    if chunk_positions and chunk_positions[-1] + chunk_duration_ms < audio_duration:
        chunk_positions.append(audio_duration - chunk_duration_ms)

    # Load Whisper model - start with medium to balance accuracy and memory
    print("Loading Whisper model...")

    # Try to use GPU if available, otherwise fall back to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Use a medium model if on CPU to improve speed
    # The base and small models are inadequate and do not detect non-English languages
    model_size = "large" if device == "cuda" else "medium"
    print(f"Using {model_size} model")

    model = whisper.load_model(model_size, device=device)

    # Initialize an empty list to store all transcriptions
    all_transcripts = []

    # Process each chunk
    print(f"Processing {len(chunk_positions)} chunks...")
    for i, start_pos in enumerate(tqdm(chunk_positions)):
        # Extract chunk
        end_pos = min(start_pos + chunk_duration_ms, audio_duration)
        chunk = audio[start_pos:end_pos]

        # Save chunk temporarily
        chunk_path = f"temp_chunks/chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")

        # Clear CUDA cache to prevent memory issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Transcribe chunk
        result = model.transcribe(chunk_path, task="transcribe")
        all_transcripts.append({
            "start": start_pos / 1000,  # Convert to seconds
            "end": end_pos / 1000,
            "text": result["text"].strip()
        })
        print(f"Chunk {i+1}/{len(chunk_positions)} transcribed")

        # Clean up temporary chunk file
        os.remove(chunk_path)

    # Clean up the temporary directory
    os.rmdir("temp_chunks")

    # Merge transcripts with overlap handling
    merged_transcript = merge_transcripts_with_overlap_handling(all_transcripts)

    return merged_transcript

# Merge the transcript chunks and remove the overlaps
def merge_transcripts_with_overlap_handling(transcripts):
    """
    Merge transcripts handling duplicated content in overlapping portions.

    Args:
        transcripts: List of dictionaries with start, end, and text fields

    Returns:
        Merged transcript
    """
    if not transcripts:
        return ""

    # Sort transcripts by start time
    transcripts = sorted(transcripts, key=lambda x: x["start"])

    # Initialize with the first transcript
    merged_text = transcripts[0]["text"]

    for i in range(1, len(transcripts)):
        current_text = transcripts[i]["text"]
        previous_text = merged_text

        # Find potential overlap in text
        overlap_found = False
        min_overlap_length = 5  # Minimum characters to consider as overlap

        for overlap_length in range(min(len(previous_text), len(current_text)), min_overlap_length - 1, -1):
            if previous_text[-overlap_length:].lower() == current_text[:overlap_length].lower():
                merged_text = previous_text + current_text[overlap_length:]
                overlap_found = True
                break

        # If no text overlap found, simply append with a space
        if not overlap_found:
            merged_text += " " + current_text

    # Clean up any multiple spaces, newlines, etc.
    merged_text = re.sub(r'\s+', ' ', merged_text).strip()

    return merged_text

# Define prompts with instructions for multilingual content
summarisation_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="""You are a professional transcription analyst skilled in multiple languages including Japanese, Hebrew, and English.

Here is a transcript of a multilingual audio recording:

{transcript}

Please carefully analyze this transcript and:

1. Identify all key points regardless of which language they appear in
2. Create a comprehensive yet concise bullet-point list of these key points
3. For each key point, maintain the original language it was spoken in
4. Ensure you don't miss important information in any language
5. Format your response as a clean, well-structured bullet list

Your output should be a multilingual summary that captures the essential content from all languages present in the recording."""
)

translation_prompt = PromptTemplate(
    input_variables=["summary", "language"],
    template="""You are an expert multilingual translator with deep cultural knowledge.

Here is a multilingual summary of key points from an audio recording:

{summary}

Please translate this entire summary into {language}, following these guidelines:

1. Translate ALL points into {language} only
2. Preserve the original meaning, tone, and nuance of each point
3. Pay special attention to cultural references, idioms, and specialised terminology
4. Maintain the bullet-point format for clarity
5. Ensure your translation is natural and fluent in {language}

Your goal is to create a translation that feels native to {language} speakers while accurately representing the original content."""
)

# Make a transcript of the audio file
def transcribe_audio_file(audio_file, transcript_file):
    """
    Transcribe audio file using chunking to avoid memory issues.

    Args:
        audio_file: Path to the audio file
        transcript_file: Path to save the transcript

    Returns:
        The transcript text
    """
    print("Starting transcription process...")
    transcript = make_transcript_in_chunks(audio_file)

    # Save the transcript
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(transcript)

    print("\nTranscript saved to file")
    print("\nFirst 500 characters of transcript:")
    print(transcript[:500] + "...")
    
    return transcript

# Summarise the content of the audio file
def summarise_and_translate(transcript, language):
    """
    Take a transcript of an audio file, summarise the content into a list of key points,
    and translate it into a language of user's choice.

    Args:
        transcript: The transcript of the audio file
        language: Target language for translation

    Returns:
        A translated list of key points in the target language
    """
    print("\nGenerating multilingual summary of key points...")
    summary = (summarisation_prompt | llm).invoke({"transcript": transcript}).content

    print("\nSummary generated. Now translating to", language)
    translated_summary = (translation_prompt | llm).invoke({"summary": summary, "language": language}).content

    # Save both versions
    with open("summary_original.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    with open(f"summary_{language}.txt", "w", encoding="utf-8") as f:
        f.write(translated_summary)

    return translated_summary

# print() helper function to display the captured output of print statements in streamlit
def capture_print_output(func, *args, **kwargs):
    # Redirect stdout to a string buffer
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = func(*args, **kwargs)
    
    # Display the captured output in Streamlit
    output_str = output.getvalue()
    if output_str:
        st.text(output_str)
    
    return result

# Streamlit UI
st.title('Audio File Summarizer')

# Welcome message
st.markdown("""
Hey there! 👋 I'm your Audio File Summarizer.  
Just share the full path to your audio file, and 
I’ll transcribe it, summarize the key points in the language of your choice, and 
save the full transcript to a file for you.
""")

# Create session state to store variables
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'audio_file'
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
if 'transcript_file' not in st.session_state:
    st.session_state.transcript_file = None
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'target_language' not in st.session_state:
    st.session_state.target_language = None
if 'final_summary' not in st.session_state:
    st.session_state.final_summary = None
if 'GPU_status' not in st.session_state:
    st.session_state.GPU_status = None

# Step 1: Get filepaths
if st.session_state.current_step == 'audio_file':
    audio_input = st.text_input("Where's your audio file? Please enter the full folder path and file name: ", key="audio_input")
    if st.button("Next", key="audio_file_button"):
        if audio_input:
            st.session_state.audio_file = audio_input
            st.spinner(f'Audio file to transcribe: {audio_input}')
            st.session_state.current_step = 'transcript_file'
            st.rerun()

if st.session_state.current_step == 'transcript_file':
    transcript_input = st.text_input("Where should we save the transcript? Please enter the full folder path and file name:", key="transcript_input")
    if st.button("Next", key="transcript_file_button"):
        if transcript_input:
            st.session_state.transcript_file = transcript_input
            st.spinner(f'Transcript file to save: {transcript_input}')
            st.session_state.current_step = 'target_language'
            st.rerun()

# Step 2: Get target language
elif st.session_state.current_step == 'target_language':
    language_input = st.text_input("In which language would you like to receive the summary?: ", key="language_input")
    if st.button("Next", key="language_button"):
        if language_input:
            st.session_state.target_language = language_input
            st.spinner(f'Summarise in: {language_input}')
            st.session_state.current_step = 'process'
            st.rerun()

# Step 3: Process audio file and produce results
elif st.session_state.current_step == 'process':

    # Display device information
    if torch.cuda.is_available():
        device_info = f"Using GPU: {torch.cuda.get_device_name(0)}"
        st.success(device_info)
    else:
        st.info("No GPU available, will use CPU (slower)")

    # Validate that we have all required data
    if not (st.session_state.audio_file and st.session_state.transcript_file and st.session_state.target_language):
        st.session_state.current_step = 'audio_file'
        st.rerun()
    try:
        # Produce a transcript of the audio file
        with st.spinner('Starting transcription...'):
            transcript = capture_print_output(
                transcribe_audio_file,
                st.session_state.audio_file,
                st.session_state.transcript_file
            )
            st.session_state.transcript = transcript
            st.success('Successfully completed transcription')
            
        # Summarise and translate
        with st.spinner('Starting Summarization and Translation...'):
            st.session_state.final_summary = capture_print_output(
                summarise_and_translate,
                st.session_state.transcript, 
                st.session_state.target_language
            )
            st.success('Successfully completed summarization and translation')
            
        # Display results    
        st.subheader("Summary of Key Points")
        st.markdown(st.session_state.final_summary)
         
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        st.error(error_msg)
        if st.button("Try again", key="error_button"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()