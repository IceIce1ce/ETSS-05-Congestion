import json

if __name__ == '__main__':
    file_path = "part_B_val.json"
    old_prefix = "/home/leeyh/Downloads/Shanghai/part_B_final/"
    new_prefix = "data/ShanghaiTech/part_B/"
    with open(file_path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array in the file")
    modified_data = [path.replace(old_prefix, new_prefix) for path in data]
    with open(file_path, 'w') as f:
        json.dump(modified_data, f, indent=4)
    print(f"Modified JSON saved to {file_path}")