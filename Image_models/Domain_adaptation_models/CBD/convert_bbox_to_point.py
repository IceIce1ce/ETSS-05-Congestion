import os
import argparse
import xml.etree.ElementTree as ET

def convert_bbox_to_points(bbox_file_path):
    with open(bbox_file_path, 'r') as file:
        lines = file.readlines()
    points = []
    bbox=[]
    for line in lines:
        x1, y1, x2, y2,l1 = map(int, line.strip().split())
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        points.append((center_y, center_x))
        bbox.append((x1, y1, x2, y2))
    return points, bbox


def create_xml_annotation(points, bboxes):
    root = ET.Element("annotation")
    for i, (point, bbox) in enumerate(zip(points, bboxes)):
        obj = ET.SubElement(root, "object")
        point_2d = ET.SubElement(obj, "point_2d")
        center_x = ET.SubElement(point_2d, "center_x")
        center_y = ET.SubElement(point_2d, "center_y")
        center_x.text = str(point[1])
        center_y.text = str(point[0])
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        ymin = ET.SubElement(bndbox, "ymin")
        xmax = ET.SubElement(bndbox, "xmax")
        ymax = ET.SubElement(bndbox, "ymax")
        xmin.text = str(bbox[0])
        ymin.text = str(bbox[1])
        xmax.text = str(bbox[2])
        ymax.text = str(bbox[3])
    return root

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for filename in os.listdir(args.input_dir):
        if filename.endswith('.txt'):
            bbox_file_path = os.path.join(args.input_dir, filename)
            points, bbox = convert_bbox_to_points(bbox_file_path)
            xml_root = create_xml_annotation(points, bbox)
            xml_file_name = filename[:-4] + '.xml'
            xml_file_path = os.path.join(args.output_dir, xml_file_name)
            tree = ET.ElementTree(xml_root)
            tree.write(xml_file_path, encoding='utf-8', xml_declaration=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='CARPK')
    parser.add_argument('--input_dir', type=str, default='data/CARPK/Annotations/')
    parser.add_argument('--output_dir', type=str, default='data/CARPK/VGG_annotation_truth/')
    args = parser.parse_args()

    print('Convert bbox to point for dataset:', args.type_dataset)
    main(args)