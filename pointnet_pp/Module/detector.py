import os
import torch
from typing import Union

from pointnet_pp.Model.cls_ssg import get_model


class Detector(object):
    def __init__(
        self, model_file_path: Union[str, None] = None, device: str = "cpu"
    ) -> None:
        self.device = device

        self.model = get_model(40, False)

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path, resume_model_only=True):
        if not os.path.exists(model_file_path):
            print("[ERROR][Detector::loadModel]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        model_dict = torch.load(
            model_file_path, map_location=torch.device(self.device))

        self.model.load_state_dict(model_dict["model_state_dict"])

        if not resume_model_only:
            # self.optimizer.load_state_dict(model_dict["optimizer"])
            self.step = model_dict["step"]
            self.eval_step = model_dict["eval_step"]
            self.loss_min = model_dict["loss_min"]
            self.eval_loss_min = model_dict["eval_loss_min"]
            self.log_folder_name = model_dict["log_folder_name"]

        print("[INFO][Detector::loadModel]")
        print("\t load model success!")
        return True

    def detect(self, points: torch.Tensor) -> torch.Tensor:
        x, l3_points = self.model(points)
        return x, l3_points
