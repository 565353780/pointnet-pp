import os
import torch
from typing import Union

from pointnet_pp.Model.cls_ssg import get_model


class Detector(object):
    def __init__(
        self, model_file_path: Union[str, None] = None, device: str = "cpu"
    ) -> None:
        self.device = device

        self.model = get_model(40, False).to(device)
        self.model.eval()

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path):
        if not os.path.exists(model_file_path):
            print("[ERROR][Detector::loadModel]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        state_dict = torch.load(
            model_file_path, map_location='cpu')

        model_state_dict = state_dict['model_state_dict']

        self.model.load_state_dict(model_state_dict)

        print("[INFO][Detector::loadModel]")
        print("\t model loaded from:", model_file_path)
        return True

    @torch.no_grad()
    def detect(self, points: torch.Tensor) -> torch.Tensor:
        x, l3_points = self.model(points)
        return x, l3_points
