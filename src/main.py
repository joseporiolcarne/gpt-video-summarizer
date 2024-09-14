import argparse
import os
from pathlib import Path
import sys
from dotenv import load_dotenv
import validators

load_dotenv()

# Get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

print(sys.path)

import time
from typing import List, Optional, Tuple
from openai import OpenAI

from pytubefix import YouTube
from pytubefix.cli import on_progress
import tiktoken
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader

from config import config
from src.utils import is_url_accessible, sanitize_filename


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def transcribe_audio(audio_file):
    """Transcribe the provided audio file using the OpenAI API with Whisper-1 model."""

    print("Starting to convert audio to transcript...")
    with open(audio_file, "rb") as audio:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=audio, response_format="srt"
        )
    return transcript


def save_transcript_as_srt(transcript, srt_file_name):
    with open(srt_file_name, "w", encoding="utf-8") as file:
        file.write(transcript)


def read_srt(srt_file_name):
    with open(srt_file_name, "r", encoding="utf-8") as file:
        return file.read()


def tokenize(text: str) -> List[str]:
    encoding = tiktoken.encoding_for_model("gpt-4-turbo")
    return encoding.encode(text)


def get_chat_completion(messages, model="gpt-4-turbo"):
    """Get a chat completion from the OpenAI API based on the provided messages."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )

    return response.choices[0].message.content


# This function chunks a text into smaller pieces based on a maximum token count and a delimiter.
def chunk_on_delimiter(input_string: str, max_tokens: int, delimiter: str) -> List[str]:
    """Chunk the input string into smaller pieces based on a maximum token count and a delimiter."""
    chunks = input_string.split(delimiter)
    combined_chunks, _, dropped_chunk_count = combine_chunks_with_no_minimum(
        chunks, max_tokens, chunk_delimiter=delimiter, add_ellipsis_for_overflow=True
    )
    if dropped_chunk_count > 0:
        print(f"warning: {dropped_chunk_count} chunks were dropped due to overflow")
    combined_chunks = [f"{chunk}{delimiter}" for chunk in combined_chunks]
    return combined_chunks


def combine_chunks_with_no_minimum(
    chunks: List[str],
    max_tokens: int,
    chunk_delimiter="\n\n",
    header: Optional[str] = None,
    add_ellipsis_for_overflow=False,
) -> Tuple[List[str], List[int]]:
    """Combine text chunks into larger blocks without exceeding a specified token count.
    It returns the combined text blocks, their original indices, and the count of chunks dropped due to overflow.
    """
    dropped_chunk_count = 0
    output = []  # list to hold the final combined chunks
    output_indices = []  # list to hold the indices of the final combined chunks
    candidate = (
        [] if header is None else [header]
    )  # list to hold the current combined chunk candidate
    candidate_indices = []
    for chunk_i, chunk in enumerate(chunks):
        chunk_with_header = [chunk] if header is None else [header, chunk]
        if len(tokenize(chunk_delimiter.join(chunk_with_header))) > max_tokens:
            print(f"warning: chunk overflow")
            if (
                add_ellipsis_for_overflow
                and len(tokenize(chunk_delimiter.join(candidate + ["..."])))
                <= max_tokens
            ):
                candidate.append("...")
                dropped_chunk_count += 1
            continue  # this case would break downstream assumptions
        # estimate token count with the current chunk added
        extended_candidate_token_count = len(
            tokenize(chunk_delimiter.join(candidate + [chunk]))
        )
        # If the token count exceeds max_tokens, add the current candidate to output and start a new candidate
        if extended_candidate_token_count > max_tokens:
            output.append(chunk_delimiter.join(candidate))
            output_indices.append(candidate_indices)
            candidate = chunk_with_header  # re-initialize candidate
            candidate_indices = [chunk_i]
        # otherwise keep extending the candidate
        else:
            candidate.append(chunk)
            candidate_indices.append(chunk_i)
    # add the remaining candidate to output if it's not empty
    if (header is not None and len(candidate) > 1) or (
        header is None and len(candidate) > 0
    ):
        output.append(chunk_delimiter.join(candidate))
        output_indices.append(candidate_indices)
    return output, output_indices, dropped_chunk_count


def summarize(
    text: str,
    detail: float = 0,
    model: str = "gpt-4-turbo",
    additional_instructions: Optional[str] = None,
    minimum_chunk_size: Optional[int] = 500,
    chunk_delimiter: str = ".",
    summarize_recursively=False,
    verbose=False,
):
    """
    Summarizes a given text by splitting it into chunks, each of which is summarized individually.
    The level of detail in the summary can be adjusted, and the process can optionally be made recursive.

    Parameters:
    - text (str): The text to be summarized.
    - detail (float, optional): A value between 0 and 1 indicating the desired level of detail in the summary.
      0 leads to a higher level summary, and 1 results in a more detailed summary. Defaults to 0.
    - model (str, optional): The model to use for generating summaries. Defaults to 'gpt-3.5-turbo'.
    - additional_instructions (Optional[str], optional): Additional instructions to provide to the model for customizing summaries.
    - minimum_chunk_size (Optional[int], optional): The minimum size for text chunks. Defaults to 500.
    - chunk_delimiter (str, optional): The delimiter used to split the text into chunks. Defaults to ".".
    - summarize_recursively (bool, optional): If True, summaries are generated recursively, using previous summaries for context.
    - verbose (bool, optional): If True, prints detailed information about the chunking process.

    Returns:
    - str: The final compiled summary of the text.

    The function first determines the number of chunks by interpolating between a minimum and a maximum chunk count based on the `detail` parameter.
    It then splits the text into chunks and summarizes each chunk. If `summarize_recursively` is True, each summary is based on the previous summaries,
    adding more context to the summarization process. The function returns a compiled summary of all chunks.
    """

    # check detail is set correctly
    assert 0 <= detail <= 1

    # interpolate the number of chunks based to get specified level of detail
    max_chunks = len(chunk_on_delimiter(text, minimum_chunk_size, chunk_delimiter))
    min_chunks = 1
    num_chunks = int(min_chunks + detail * (max_chunks - min_chunks))

    # adjust chunk_size based on interpolated number of chunks
    document_length = len(tokenize(text))
    chunk_size = max(minimum_chunk_size, document_length // num_chunks)
    text_chunks = chunk_on_delimiter(text, chunk_size, chunk_delimiter)
    if verbose:
        print(f"Splitting the text into {len(text_chunks)} chunks to be summarized.")
        print(f"Chunk lengths are {[len(tokenize(x)) for x in text_chunks]}")

    # set system message
    system_message_content = "Rewrite this text in summarized form."
    if additional_instructions is not None:
        system_message_content += f"\n\n{additional_instructions}"

    accumulated_summaries = []
    for chunk in tqdm(text_chunks):
        if summarize_recursively and accumulated_summaries:
            # Creating a structured prompt for recursive summarization
            accumulated_summaries_string = "\n\n".join(accumulated_summaries)
            user_message_content = f"Previous summaries:\n\n{accumulated_summaries_string}\n\nText to summarize next:\n\n{chunk}"
        else:
            # Directly passing the chunk for summarization without recursive context
            user_message_content = chunk

        # Constructing messages based on whether recursive summarization is applied
        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content},
        ]

        # Assuming this function gets the completion and works as expected
        response = get_chat_completion(messages, model=model)
        accumulated_summaries.append(response)

    # Compile final summary from partial summaries
    final_summary = "\n\n".join(accumulated_summaries)

    return final_summary


def main():
    parser = argparse.ArgumentParser(description="Video Summarizer")
    parser.add_argument(
        "input_source",
        type=str,
        nargs="?",
        help="URL of the YouTube video or path to a local MP3/MP4 file to summarize",
    )

    parser.add_argument(
        "--detail",
        type=float,
        default=config.SUMMARIZE_DEFAULT_DETAIL,
        help=f"Level of detail for the summary (default: {config.SUMMARIZE_DEFAULT_DETAIL})",
    )
    parser.add_argument(
        "--use-subtitles",
        type=bool,
        default=True,
        help="Use subtitles if available (default: True)",
    )

    parser.add_argument(
        "--language",
        type=str,
        default=config.SUMMARIZE_DEFAULT_LANGUAGE,
        help=f"Language of the summary (default: {config.SUMMARIZE_DEFAULT_LANGUAGE})",
    )
    # List available templates excluding base_template.jinja2
    available_templates = [
        f.rsplit(".", 1)[0]
        for f in os.listdir(config.TEMPLATE_DIR)
        if f.endswith(".jinja2") and f != "base_template.jinja2"
    ]

    parser.add_argument(
        "--template",
        type=str,
        default=config.SUMMARIZE_DEFAULT_TEMPLATE,
        help=f"Template to use for summarization (default: markdown). Available templates: {', '.join(available_templates)}",
    )

    parser.add_argument(
        "--careful-words",
        type=str,
        default=config.SUMMARIZE_CAREFUL_WORDS,
        help="Words to be careful with in the summary",  # eg. "careful, words, here"
    )

    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Print detailed information about the process (default: False)",
    )

    args = parser.parse_args()

    input_source = args.input_source

    if not input_source:
        input_source = input("Enter the video URL or local file path: ")

    use_subtitles = args.use_subtitles
    language = args.language
    careful_words = args.careful_words
    verbose = args.verbose

    # Convert TEMPORARY_DIR to a Path object if it's not already
    tmpdirname = Path(config.TEMPORARY_DIR)

    # Check if the directory exists; if not, create it
    if not tmpdirname.exists():
        tmpdirname.mkdir(parents=True, exist_ok=True)

    # Iterate through the files in the directory and delete them
    for file in tmpdirname.iterdir():
        try:
            if file.is_file():  # Check if the path is a file before trying to delete
                file.unlink()
            elif file.is_dir():  # Optionally handle directories if needed
                file.rmdir()  # Use with caution; this removes empty directories only
        except Exception as e:
            print(f"An error occurred while deleting {file}: {e}")

    save_path = tmpdirname

    if validators.url(input_source):
        try:
            if is_url_accessible(input_source):
                print("URL is valid and is accessible.")
            print(f"Downloading video to project directory: {tmpdirname}")
            yt = YouTube(input_source, on_progress_callback=on_progress)
            print(f"Checking for English subtitles for: {yt.title}")
            if use_subtitles:
                print("Subtitles will be used if available.")
            else:
                print("Subtitles will not be used.")
            caption = yt.captions.get_by_language_code("en") if use_subtitles else None
            if caption:
                print("English subtitles found. Downloading subtitles...")
                srt_filename = tmpdirname / "captions.srt"
                caption.save_captions(srt_filename)
                transcript = read_srt(srt_filename)
            else:
                print(
                    "No English subtitles found or subtitles not used. Downloading audio..."
                )
                audio_stream = yt.streams.get_audio_only()
                if not audio_stream:
                    raise Exception("No valid audio stream found.")

                downloaded_file = audio_stream.download(output_path=str(save_path))
                base, ext = os.path.splitext(downloaded_file)
                new_file = base + ".mp3"
                os.rename(downloaded_file, new_file)
                transcript = transcribe_audio(new_file)

            # Create a descriptive filename for the transcript
            video_title = yt.title.replace(" ", "_").replace("/", "_")
            video_title = sanitize_filename(video_title)

            srt_filename = os.path.join(
                config.OUTPUT_DIR, f"{video_title}_transcript.srt"
            )
            save_transcript_as_srt(transcript, srt_filename)
            print(f"Successfully saved transcript as {srt_filename}.")
        except Exception as e:
            print(f"An error occurred while downloading the video: {e}")
            sys.exit(1)
    elif os.path.isfile(input_source) and input_source.lower().endswith(
        (".mp3", ".mp4")
    ):
        print(f"Processing local file: {input_source}")
        if input_source.lower().endswith(".mp3"):
            transcript = transcribe_audio(input_source)
        elif input_source.lower().endswith(".mp4"):
            # Extract audio from mp4 and transcribe
            print("Extracting audio from MP4...")
            audio_file = os.path.join(tmpdirname, "extracted_audio.mp3")
            os.system(f"ffmpeg -i {input_source} -q:a 0 -map a {audio_file}")
            transcript = transcribe_audio(audio_file)

        # Create a descriptive filename for the transcript
        video_title = os.path.splitext(os.path.basename(input_source))[0]
        video_title = sanitize_filename(video_title)

        srt_filename = os.path.join(config.OUTPUT_DIR, f"{video_title}_transcript.srt")
        save_transcript_as_srt(transcript, srt_filename)
        print(f"Successfully saved transcript as {srt_filename}.")
    else:
        print("Error: Input must be a valid YouTube URL or a local MP3/MP4 file.")
        sys.exit(1)

    srt_content = read_srt(srt_filename)
    print("Starting to generate summary...")

    # Load Jinja2 environment
    # Set the loader with the absolute path to the templates folder
    env = Environment(
        loader=FileSystemLoader(os.path.join(config.PROJECT_DIR, "templates"))
    )

    template = env.get_template(f"{args.template}.jinja2")

    # Render the selected template
    additional_instructions = template.render(
        language=language, careful_words=careful_words
    )
    summary = summarize(
        srt_content,
        detail=args.detail,
        additional_instructions=additional_instructions,
        verbose=verbose,
    )

    if not summary.strip():
        raise ValueError("The summary is empty. Please check the input and try again.")

    # Save the summary to the clipboard
    try:
        import pyperclip

        pyperclip.copy(summary)
        print("Summary copied to clipboard.")
    except Exception as e:
        print(f"An error occurred while copying the summary to the clipboard: {e}")

    print("Summary:\n", summary)

    # Create a descriptive filename for the summary
    summary_filename = f"{video_title}_summary_{int(time.time())}.txt"
    with open(
        os.path.join(config.OUTPUT_DIR, summary_filename), "w", encoding="utf-8"
    ) as file:
        file.write(summary)

    print(f"Successfully saved summary as {summary_filename}.")

    print("Cleaning up temporary files...")

    for file in tmpdirname.iterdir():
        try:
            file.unlink()
        except Exception as e:
            print(f"An error occurred while deleting {file}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
