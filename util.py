import json

def parse_json(json_file_path: str) -> dict:
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
        return {}