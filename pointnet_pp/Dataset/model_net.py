import os
import pickle
import warnings
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from pointnet_pp.Method.point import pc_normalize, farthest_point_sample_np

warnings.filterwarnings("ignore")


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split="train", process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, "modelnet10_shape_names.txt")
        else:
            self.catfile = os.path.join(self.root, "modelnet40_shape_names.txt")

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids["train"] = [
                line.rstrip()
                for line in open(os.path.join(self.root, "modelnet10_train.txt"))
            ]
            shape_ids["test"] = [
                line.rstrip()
                for line in open(os.path.join(self.root, "modelnet10_test.txt"))
            ]
        else:
            shape_ids["train"] = [
                line.rstrip()
                for line in open(os.path.join(self.root, "modelnet40_train.txt"))
            ]
            shape_ids["test"] = [
                line.rstrip()
                for line in open(os.path.join(self.root, "modelnet40_test.txt"))
            ]

        assert split == "train" or split == "test"
        shape_names = ["_".join(x.split("_")[0:-1]) for x in shape_ids[split]]
        self.datapath = [
            (
                shape_names[i],
                os.path.join(self.root, shape_names[i], shape_ids[split][i]) + ".txt",
            )
            for i in range(len(shape_ids[split]))
        ]
        print("The size of %s data is %d" % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(
                root,
                "modelnet%d_%s_%dpts_fps.dat"
                % (self.num_category, split, self.npoints),
            )
        else:
            self.save_path = os.path.join(
                root,
                "modelnet%d_%s_%dpts.dat" % (self.num_category, split, self.npoints),
            )

        if self.process_data:
            if not os.path.exists(self.save_path):
                print(
                    "Processing data %s (only running in the first time)..."
                    % self.save_path
                )
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=",").astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample_np(point_set, self.npoints)
                    else:
                        point_set = point_set[0 : self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, "wb") as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print("Load processed data from %s..." % self.save_path)
                with open(self.save_path, "rb") as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=",").astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample_np(point_set, self.npoints)
            else:
                point_set = point_set[0 : self.npoints, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == "__main__":
    import torch

    data = ModelNetDataLoader("/data/modelnet40_normal_resampled/", split="train")
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)