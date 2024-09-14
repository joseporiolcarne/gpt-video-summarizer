
# 🎥 GPT-VIDEO-SUMMARIZER: Your Video Summarization Companion

## 🌟 Description

Welcome to Gpt-Video-Summarizer, a tool for transforming lengthy video content into concise, digestible summaries. Whether you're a student, researcher, or content creator, this tool streamlines the process by leveraging cutting-edge AI models. It utilizes existing YouTube subtitles or extracts audio from videos, transcribing it into an SRT file using OpenAI's Whisper model, and then crafts a summary with GPT-4 Turbo.

## 🚀 Features
- **Audio Extraction**: Seamlessly extracts audio from youtube or mp4 videos of any length,
- **AI-Powered Transcription**: Converts audio to text using OpenAI's Whisper model with your own api key.
- **Intelligent Summarization**: Generates insightful summaries with OpenAI's GPT model.
- **Flexible Output**: Save transcripts and summaries in various formats.
- **Clipboard Integration**: Instantly copy summaries for quick access.

## 📋 Requirements
- **Python**: Version 3.8 or later.
- **Libraries**: `moviepy`, `openai`, `python-dotenv`, `pytubefix`, `validators`, `pyperclip`.

## 🛠️ Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   cp .env.example .env   
   ```

2. **Add OpenAI API Key**:

   Add your OpenAI API key to the `.env` file.

3. **Install it as package (Optional)**:

   ```bash
   pip install .   
   ```


## 🎯 Usage

1. **Run the Script**: Pass the URL to the script or follow the interactive prompt.
   ```bash
   gpt-video-summarize https://www.youtube.com/watch?v=MkTw3_PmKtc
   ```

2. **Alternative Command with Parameters:**


 ```bash
   gpt-video-summarize --video-url <video-url> --detail 0.1 --language English --template markdown
```

You can also set preferences in config.py and run the script with

```bash
   python -m src/main.py
```


## 📥 Input Parameters
- **`video-url`**: The URL or local path of the video you wish to summarize.
- **`detail`**: [0,1]  One means more detailed.
- **`language`**: Desired language for the summary.
- **`template`**: Choose a template for summarization. ["Text", "Markdown", "Obsidian", "CSV"]
- **`careful-words`**: Specify words to handle with care in the transcript.

## 📤 Output Files

- **Transcript**: Saved as `SRT`.
- **Summary**: Saved as `TXT`.

## Example output

### Example Output

**Transcript (SRT Format):**

```arduino
1
00:00:00,000 --> 00:00:02,000
This is an example transcript line.
```
**Summary (TXT Format):**

```css
## Summary

- Main point 1...
- Main point 2...
```

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.

## 🙏 Acknowledgments
- **OpenAI**: For the Whisper and GPT models.
- **Community**: Developers of `moviepy` and `pyperclip` for their invaluable libraries.
