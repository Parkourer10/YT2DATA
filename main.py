import yt_dlp
import whisper
import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import json

# Template for question and answer generation
template = """
Create a question-answer pair based on the following transcription.

Transcription: {transcription}

Question:
Answer:
"""

model = OllamaLLM(model='llama3.2:3b')
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model  

def download_youtube_video(url):
    download_path = os.getcwd()  
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{download_path}/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        title = info_dict.get('title', None)
        ydl.download([url])
    return f"{download_path}/{title}.mp3", title

youtube_url = input('Enter YouTube URL: ')
mp3_path, video_title = download_youtube_video(youtube_url)
print(f'File saved at: {mp3_path}')

model = whisper.load_model("small")  # Whisper models include tiny, base, small, medium, large, turbo

def transcribe_audio(mp3_path):
    result = model.transcribe(mp3_path)
    return result['text']

def get_unique_filename(base_path):
    counter = 1
    file_path = f"{base_path[:-4]}.txt"
    while os.path.exists(file_path):
        file_path = f"{base_path[:-4]}{counter}.txt"
        counter += 1
    return file_path

def save_transcription_to_file(text, output_file):
    unique_output_file = get_unique_filename(output_file)
    with open(unique_output_file, 'w') as file:
        file.write(text)
    return unique_output_file

def transcribe():
    mp3_file_path = mp3_path
    base_output_file = f'{video_title}.txt'  

    if not os.path.isfile(mp3_file_path):
        print(f"Error: The file {mp3_file_path} does not exist.")
        return

    print(f"Transcribing {mp3_file_path}...")
    transcription = transcribe_audio(mp3_file_path)

    print(f"Transcription complete. Saving...")
    output_text_file = save_transcription_to_file(transcription, base_output_file)

    print(f"Transcription saved successfully to: {output_text_file}")
    return output_text_file  

def txt2dataset(output_text_file):
    
    with open(output_text_file, 'r') as file:
        transcription = file.read()

    # Split transcription into batches of 2048 words
    words = transcription.split() 
    batches = []
    current_batch = []

    for word in words:
        current_batch.append(word)
        
        if len(current_batch) >= 2048: 
            batches.append(' '.join(current_batch))
            current_batch = [] 
    
    
    if current_batch:
        batches.append(' '.join(current_batch))

    dataset = []

    
    for section in batches:
        section = section.strip()
        if len(section) > 0:
            qa_pair = generate_qa_pair(section)
            dataset.append(qa_pair)

    
    dataset_file = output_text_file.replace('.txt', '_qa_dataset.json')
    with open(dataset_file, 'w') as json_file:
        json.dump(dataset, json_file, indent=4)

    print(f"Question-Answer dataset saved to: {dataset_file}")

def generate_qa_pair(section):
    result = chain.invoke({"transcription": section})
    return result

if __name__ == "__main__":
    file = transcribe() 
    if file:
        txt2dataset(file)  
       
