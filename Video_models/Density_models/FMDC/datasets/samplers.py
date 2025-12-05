import torch
import  random

class ColorizationSampler():
    def __init__(self, labels, frame_intervals):
        self.frame_intervals = frame_intervals
        self.n_sample = len(labels)
        self.scene_id = {}
        self.scenes = []
        for idx, label in enumerate(labels):
            scene_name = label['scene_name']
            if scene_name not in self.scene_id.keys():
                self.scene_id.update({scene_name:[idx,0]})
            self.scene_id[scene_name][1]+=1
            self.scenes.append(scene_name)
        self.current_scene = random.choice(list(self.scene_id.keys()))
        self.current_scene_frame = random.randint(0, 10)

    def __len__(self):
        return self.n_sample

    def __iter__(self):
        for i in range(self.n_sample - 1):
            batch = []
            tmp_intervals = random.randint(self.frame_intervals[0], min(self.scene_id[self.current_scene][1] // 2,self.frame_intervals[1]))
            if self.current_scene_frame + tmp_intervals <= self.scene_id[self.current_scene][1]:
                c = self.scene_id[self.current_scene][0] + self.current_scene_frame
                pair_c = c + tmp_intervals
                self.current_scene_frame += tmp_intervals
            else:
                self.current_scene = random.choice(list(self.scene_id.keys()))
                self.current_scene_frame = random.randint(0,10)
                c = self.current_scene_frame + self.scene_id[self.current_scene][0]
                tmp_intervals = random.randint(self.frame_intervals[0], min(self.scene_id[self.current_scene][1] // 2,self.frame_intervals[1]))
                pair_c = c + tmp_intervals
                self.current_scene_frame += tmp_intervals
                assert self.scenes[c] == self.scenes[pair_c]
            batch.append(torch.tensor([c, pair_c]))
            batch = torch.stack(batch).reshape(-1)
            yield batch

class CategoriesSampler():
    def __init__(self, labels, frame_intervals, n_per):
        self.frame_intervals = frame_intervals
        self.n_sample = len(labels)
        self.n_batch = self.n_sample// n_per
        self.n_per = n_per
        self.scenes = []
        self.scene_id = {}
        for idx, label in enumerate(labels):
            scene_name = label['scene_name']
            if scene_name not in self.scene_id.keys():
                self.scene_id.update({scene_name:0})
            self.scene_id[scene_name] += 1
            self.scenes.append(scene_name)

    def __len__(self):
        return self.n_sample

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            frame_a = torch.randperm(self.n_sample)[:self.n_per]
            for c in frame_a:
                scene_name = self.scenes[c]
                tmp_intervals = random.randint(self.frame_intervals[0], min(self.scene_id[scene_name] // 2,self.frame_intervals[1]))
                if c<self.n_sample-tmp_intervals:
                    if self.scenes[c + tmp_intervals] == scene_name:
                        pair_c = c + tmp_intervals
                    else:
                        pair_c = c
                        c = c - tmp_intervals
                else:
                    pair_c = c
                    c = c - tmp_intervals
                assert self.scenes[c] == self.scenes[pair_c]
                batch.append(torch.tensor([c, pair_c]))
            batch = torch.stack(batch).reshape(-1)
            yield batch