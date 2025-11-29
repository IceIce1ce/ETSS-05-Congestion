import os
from PIL import Image

if __name__ == '__main__':
    video_dir = "data/Sense/video_ori"
    label_dir = "data/Sense/label_list_all"
    processed_dir = "data/Sense/processed_labels"
    os.makedirs(processed_dir, exist_ok=True)
    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue
        input_path = os.path.join(label_dir, label_file)
        output_path = os.path.join(processed_dir, label_file)
        folder_name = os.path.splitext(label_file)[0]
        image_folder = os.path.join(video_dir, folder_name)
        with open(input_path, "r") as fin, open(output_path, "w") as fout:
            for line in fin:
                parts = line.strip().split()
                if not parts:
                    continue
                frame_id = parts[0]
                image_name = frame_id + ".jpg"
                image_path = os.path.join(image_folder, image_name)
                with Image.open(image_path) as img:
                    width, height = img.size
                annots = parts[1:]
                new_entries = []
                for i in range(0, len(annots), 7):
                    head_id, x1, y1, x2, y2, p1, p2 = annots[i:i+7]
                    new_entries.append(f"{x1} {y1} {x2} {y2} {p1} {p2} {head_id}")
                new_line = f"{image_name} {width} {height} " + " ".join(new_entries)
                fout.write(new_line + "\n")
    print("✅ Processing complete! Saved in", processed_dir)