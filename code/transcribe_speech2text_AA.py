import argparse
import glob
import json
from pathlib import Path

import assemblyai as aai


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api_key",
        required=True,
        type=str,
        help="AssemblyAI API token",
    )
    return parser.parse_args()


def transcribe_speech2text() -> None:
    """."""
    wf_list = [Path(x).resolve() for x in sorted(
        glob.glob("../sourcedata/*/*wav")
    )]
    for w_file in wf_list:
        language = w_file.parent.name
        config = aai.TranscriptionConfig(
            speech_models=["universal-3-pro", "universal-2"],
            language_code=language.lower(),
        )    
        out_name = Path(w_file).name.replace(
            "section_", "section-"
        ).replace(".wav", "_model-AA_transcript.json")
        out_path = Path(
            f"../annotations/transcripts/{language}/{out_name}"
        ).resolve()
        if not Path(out_path).exists():
            transcript = aai.Transcriber(config=config).transcribe(str(w_file))

            json_transcript = {
                "transcript": transcript.text,
                "words": [
                    {
                        "word": x.text,
                        "start": float(x.start)/1000,
                        "end": float(x.end)/1000,
                        "confidence": x.confidence
                    } for x in transcript.words],
            }
            with open(out_path, "w") as outfile:
                json.dump(json_transcript, outfile, indent=4)


if __name__ == "__main__":
    """."""
    args = get_arguments()
    aai.settings.api_key = args.api_key

    transcribe_speech2text()