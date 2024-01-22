import torch

from pointnet_pp.Module.detector import Detector


def demo():
    model_file_path = "./pretrained/cls_ssg/best_model.pth"
    device = "cpu"

    points = torch.rand(2, 3, 4000).type(torch.float32).to(device)

    detector = Detector(model_file_path, device)
    _, features = detector.detect(points)
    print(features.shape)
    return True
