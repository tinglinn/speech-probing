import argparse
import json
import os


def create_transcript(json_path):
    with open(json_path) as f:
        for jsonObj in f:
            audio_info = json.load(jsonObj)
            filepath = audio_info["key"]
            transcript_path = os.path.join(filepath.split(".")[0],".txt")
            text = audio_info["text"]
            open("transcript_path", "w").write(text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', type=str,
                        help='Path to json file')
    args = parser.parse_args()
    create_transcript(args.json_path)