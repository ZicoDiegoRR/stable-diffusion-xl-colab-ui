from diffusers.utils import load_image, make_image_grid
from controlnet_aux import OpenposeDetector
from transformers import pipeline as pipe
from PIL import Image as ImagePIL
import numpy as np
import torch
import cv2

class ControlNetImage:
    # Return a Canny edge detection image
    def get_canny(self, img):
        image_canny = cv2.Canny(np.array(img), minimum_canny_threshold, maximum_canny_threshold)
        image_canny = image_canny[:, :, None]
        image_canny = np.concatenate([image_canny, image_canny, image_canny], axis=2)
        canny_image = ImagePIL.fromarray(image_canny)
        return canny_image

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
        return self.openpose_estimator(image)

    # Initiate two image estimators
    def load_pipe(self):
        self.depth_estimator = pipe("depth-estimation", device="cpu")
        self.openpose_estimator = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to("cpu")
