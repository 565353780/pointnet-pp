import torch
from tqdm import trange

from pointnet_pp.Module.detector import Detector


def demo():
    model_file_path = "../pointnet-pp/pretrained/cls_ssg/best_model.pth"
    device = "cpu"

    points = torch.rand(1, 3, 4000).type(torch.float32).to(device)

    detector = Detector(model_file_path, device)
    for _ in trange(100):
        _, features = detector.detect(points)
    print(features.shape)
    return True
