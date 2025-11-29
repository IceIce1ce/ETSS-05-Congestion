import os

if __name__ == '__main__':
    txt_file = "scene_label.txt"
    video_dir = "data/Sense/video_ori"
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    folders = sorted([f for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))])
    if len(folders) != len(lines):
        print(f"Error: Number of folders ({len(folders)}) does not match number of lines ({len(lines)})")
        exit()
    new_lines = []
    for i, line in enumerate(lines):
        columns = line.strip().split()
        if columns:
            columns[0] = folders[i]
            new_lines.append(" ".join(columns) + "\n")
    with open(txt_file, 'w') as f:
        f.writelines(new_lines)
    print("Replacement completed successfully!")