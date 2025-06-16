from diffusers.utils import load_image, make_image_grid
from controlnet_aux import OpenposeDetector
from transformers import pipeline as pipe
from PIL import Image as ImagePIL
import numpy as np
import torch
import cv2

class ControlNetImage:
    # Return a Canny edge detection image
    def get_canny(self, pil_image, minimum_canny_threshold, maximum_canny_threshold):
        image = np.array(pil_image)
        image = cv2.Canny(image, minimum_canny_threshold, maximum_canny_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = ImagePIL.fromarray(image)
        return image

    # Return a Depth Map image
    def get_depth(self, img, output="get"):
        depth_image = self.depth_estimator(img)["depth"]
        depth_image = np.array(depth_image)
        depth_image = depth_image[:, :, None]
        depth_image = np.concatenate([depth_image, depth_image, depth_image], axis=2)
        if output == "display":
            return ImagePIL.fromarray(depth_image) # For display

        detected_map = torch.from_numpy(depth_image).float() / 255.0
        depth_map = detected_map.permute(2, 0, 1)
        return depth_map

        
    # Return an Open Pose image
    def get_openpose(self, image):
        openpose_image = self.openpose_estimator(image)
        return openpose_image

    # Initiate two image estimators
    def load_pipe(self):
        self.depth_estimator = pipe("depth-estimation", device="cpu")
        self.openpose_estimator = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to("cpu")
