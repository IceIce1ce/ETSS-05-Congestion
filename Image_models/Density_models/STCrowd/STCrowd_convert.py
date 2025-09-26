import os
import json
from concurrent import futures as futures
import mmengine
import argparse
import numpy as np

def load_file(path_root,load_dict):
    def process_single_scene(idx):
        info = {}
        pc_info = {'num_features': 4}
        image_info = {'image_idx': idx, 'image_path': path_root + '/'.join(load_dict['frames'][idx]['images'][0]['image_name'].split('/')[-4:])}
        pc_info['point_cloud_path'] = path_root + '/'.join(load_dict['frames'][idx]['frame_name'].split('/')[-4:])
        info['image'] = image_info
        info['point_cloud'] = pc_info
        position_list, boundingbox_list, occlusion_list, rotation_list, img_point_list = [], [], [], [], []
        v_list = []
        tracking_id_list = []
        if len(load_dict['frames'][idx]['items']) == 0:
            position_list.append([0.1, 0.1, 0.1])
            boundingbox_list.append([0.001, 0.001, 0.001])
            occlusion_list.append(0)
            rotation_list.append(0)
            v_list.append([0, 0])
            tracking_id_list.append(0)
        for item in load_dict['frames'][idx]['items']:
            if item['boundingbox']['z'] < 1.2: # delete the sitting person currently
                continue
            position_list.append([item['position']['x'], item['position']['y'], item['position']['z']])
            boundingbox_list.append([item['boundingbox']['x'],item['boundingbox']['y'],item['boundingbox']['z']])
            occlusion_list.append(item['occlusion'])
            rotation_list.append(item['rotation'])
            vx = 0
            vy = 0
            try:
                if idx != 0:
                    next_items_id_list = [next_frame_item['id'] for next_frame_item in load_dict['frames'][idx-1]['items']]
                    for j in range(len(next_items_id_list)):
                        if item['id'] == next_items_id_list[j]:
                            vx =  load_dict['frames'][idx-1]['items'][j]['position']['x'] - item['position']['x']
                            vy = load_dict['frames'][idx-1]['items'][j]['position']['y'] - item['position']['y']
                            break
            except:
                vx = 0
                vy = 0
            v_list.append([vx,vy])
            tracking_id_list.append(item['id'])
        img_id, img_bbox, img_occ= [], [] ,[]
        for item in load_dict['frames'][idx]['images'][0]['items']:
            eight_point = []
            for p in item['points']:
                eight_point.append(p['x'])
                eight_point.append(p['y'])
            img_id.append(item['id'])
            img_occ.append(item['occlusion'])
            img_point_list.append(np.asarray(eight_point).reshape(8, 2))
            img_bbox.append([item['boundingbox']['x'], item['boundingbox']['y'], item['dimension']['x'], item['dimension']['y']])
        info['annos'] = {'position': position_list, 'dimensions': boundingbox_list, 'occlusion': occlusion_list, 'rotation': rotation_list, 'tracking_id': tracking_id_list, 'image_bbox': dict()}
        info['annos']['image_bbox']['3D'] = img_point_list
        info['annos']['image_bbox']['2D'] = img_bbox
        info['annos']['image_bbox']['occlusion'] = img_occ
        info['annos']['image_id'] = img_id
        info_tracking = {'group_id': load_dict['group_id'],'indx':idx,'velocity': v_list,'item_id':tracking_id_list}
        info['annos']['tracking'] = info_tracking
        return info
    ids = list(range(load_dict['total_number']))
    with futures.ThreadPoolExecutor(1) as executor:
        infos = executor.map(process_single_scene, ids)
    return list(infos)

def create_data_info(data_path, file_list):
    path = os.path.join(data_path, 'anno/')
    all_files = [str(file) + '.json' for file in file_list]
    infos = []
    for file in all_files:
        file_group_path = ''.join([path, file])
        with open(file_group_path, 'r') as load_f:
            load_dict = json.load(load_f)
        info = load_file(data_path, load_dict)
        if info:
            infos = infos + info
    return infos

def main(args):
    split_file = os.path.join(args.input_dir, args.split_file)
    with open(split_file, 'r') as load_f:
        load_dict = json.load(load_f)
    info = create_data_info(data_path=args.input_dir, file_list=load_dict[args.split])
    filename = os.path.join(args.input_dir, f'STCrowd_infos_{args.split}.pkl')
    mmengine.dump(info, filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='STCrowd')
    parser.add_argument('--input_dir', type=str, default='datasets/STCrowd')
    parser.add_argument('--split_file', type=str, default='split.json')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])
    args = parser.parse_args()

    print('Process dataset {} with split {}'.format(args.type_dataset, args.split))
    main(args)