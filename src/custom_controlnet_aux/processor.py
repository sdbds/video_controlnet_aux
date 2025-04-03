"""
This file contains a Processor that can be used to process images with controlnet aux processors
"""
import io
import logging
from typing import Dict, Optional, Union

from PIL import Image

from .__init__ import (
    AnimalposeDetector,
    AnimeFaceSegmentor,
    BinaryDetector,
    CannyDetector,
    ColorDetector,
    ColorShuffleDetector,
    ContentShuffleDetector,
    DenseposeDetector,
    DownSampleDetector,
    DwposeDetector,
    GrayDetector,
    HEDdetector,
    Image2MaskShuffleDetector,
    LeresDetector,
    LineartDetector,
    LineartAnimeDetector,
    LineartMangaDetector,
    MediapipeFaceDetector,
    MidasDetector,
    MLSDdetector,
    NormalBaeDetector,
    OneformerSegmentor,
    OpenposeDetector,
    PidiNetDetector,
    SamDetector,
    ScribbleDetector,
    ScribbleXDog_Detector,
    TileDetector,
    UniformerSegmentor,
    ZoeDetector,
)

from .util import get_torch_device

LOGGER = logging.getLogger(__name__)

DEVICE = get_torch_device()

MODELS = {
    # checkpoint models
    "scribble_hed": {"class": HEDdetector, "checkpoint": True},
    "softedge_hed": {"class": HEDdetector, "checkpoint": True},
    "scribble_hedsafe": {"class": HEDdetector, "checkpoint": True},
    "softedge_hedsafe": {"class": HEDdetector, "checkpoint": True},
    "depth_midas": {"class": MidasDetector, "checkpoint": True},
    "mlsd": {"class": MLSDdetector, "checkpoint": True},
    "openpose": {"class": OpenposeDetector, "checkpoint": True},
    "openpose_face": {"class": OpenposeDetector, "checkpoint": True},
    "openpose_faceonly": {"class": OpenposeDetector, "checkpoint": True},
    "openpose_full": {"class": OpenposeDetector, "checkpoint": True},
    "openpose_hand": {"class": OpenposeDetector, "checkpoint": True},
    "scribble_pidinet": {"class": PidiNetDetector, "checkpoint": True},
    "softedge_pidinet": {"class": PidiNetDetector, "checkpoint": True},
    "scribble_pidsafe": {"class": PidiNetDetector, "checkpoint": True},
    "softedge_pidsafe": {"class": PidiNetDetector, "checkpoint": True},
    "normal_bae": {"class": NormalBaeDetector, "checkpoint": True},
    "normal_midas": {"class": MidasDetector,"depth_and_normal": True},
    "lineart_coarse": {"class": LineartDetector, "checkpoint": True},
    "lineart_realistic": {"class": LineartDetector, "checkpoint": True},
    "lineart_anime": {"class": LineartAnimeDetector, "checkpoint": True},
    "lineart_anime_denoise": {"class": LineartMangaDetector, "checkpoint": True},
    "depth_zoe": {"class": ZoeDetector, "checkpoint": True},
    "depth_leres": {"class": LeresDetector, "checkpoint": True},
    "depth_leres++": {"class": LeresDetector, "checkpoint": True},
    "anime_face_segment": {"class": AnimeFaceSegmentor, "checkpoint": True},
    "densepose": {"class": DenseposeDetector, "checkpoint": True},
    "densepose_normal": {"class": DenseposeDetector, "checkpoint": True},
    "dw_openpose": {
        "class": DwposeDetector,
        "checkpoint": True,
        "torchscript_device": True,
    },
    "dw_openpose_face": {
        "class": DwposeDetector,
        "checkpoint": True,
        "torchscript_device": True,
    },
    "dw_openpose_faceonly": {
        "class": DwposeDetector,
        "checkpoint": True,
        "torchscript_device": True,
    },
    "dw_openpose_hand": {
        "class": DwposeDetector,
        "checkpoint": True,
        "torchscript_device": True,
    },
    "dw_openpose_full": {
        "class": DwposeDetector,
        "checkpoint": True,
        "torchscript_device": True,
    },
    "animal_openpose": {
        "class": AnimalposeDetector,
        "checkpoint": True,
        "torchscript_device": True,
    },
    "uniformer_ufade20k": {"class": UniformerSegmentor, "checkpoint": True},
    "oneformer_coco": {
        "class": OneformerSegmentor,
        "checkpoint": True,
        "filename": "150_16_swin_l_oneformer_coco_100ep.pth",
    },
    "oneformer_ade20k": {"class": OneformerSegmentor, "checkpoint": True},
    "scribble_xdog": {"class": ScribbleXDog_Detector, "checkpoint": True},
    "scribble": {"class": ScribbleDetector, "checkpoint": True},
    "sam": {"class": SamDetector, "checkpoint": True},
    # instantiate
    "shuffle": {"class": ContentShuffleDetector, "checkpoint": False},
    "mediapipe_face": {"class": MediapipeFaceDetector, "checkpoint": False},
    "canny": {"class": CannyDetector, "checkpoint": False},
    "tile": {"class": TileDetector, "checkpoint": False},
    "binary": {"class": BinaryDetector, "checkpoint": False},
    "color": {"class": ColorDetector, "checkpoint": False},
}


MODEL_PARAMS = {
    "scribble_hed": {"scribble": True},
    "softedge_hed": {"scribble": False},
    "scribble_hedsafe": {"scribble": True, "safe": True},
    "softedge_hedsafe": {"scribble": False, "safe": True},
    "depth_midas": {},
    "mlsd": {},
    "openpose": {"include_body": True, "include_hand": False, "include_face": False},
    "openpose_face": {
        "include_body": True,
        "include_hand": False,
        "include_face": True,
    },
    "openpose_faceonly": {
        "include_body": False,
        "include_hand": False,
        "include_face": True,
    },
    "openpose_full": {"include_body": True, "include_hand": True, "include_face": True},
    "dw_openpose": {
        "include_body": True,
        "include_hand": False,
        "include_face": False,
    },
    "dw_openpose_face": {
        "include_body": True,
        "include_hand": False,
        "include_face": True,
    },
    "dw_openpose_faceonly": {
        "include_body": False,
        "include_hand": False,
        "include_face": True,
    },
    "dw_openpose_full": {
        "include_body": True,
        "include_hand": True,
        "include_face": True,
    },
    "dw_openpose_hand": {
        "include_body": False,
        "include_hand": True,
        "include_face": False,
    },
    "animal_openpose": {},
    "anime_face_segment": {},
    "scribble_pidinet": {"safe": False, "scribble": True},
    "softedge_pidinet": {"safe": False, "scribble": False},
    "scribble_pidsafe": {"safe": True, "scribble": True},
    "softedge_pidsafe": {"safe": True, "scribble": False},
    "densepose": {},
    "densepose_normal": {"cmap": "nroaml"},
    "uniformer_ufade20k": {},
    "oneformer_coco": {},
    "oneformer_ade20k": {},
    "scribble_xdog": {},
    "scribble": {},
    "sam": {},
    "normal_bae": {},
    "normal_midas": {"depth_and_normal": True},
    "lineart_realistic": {"coarse": False},
    "lineart_coarse": {"coarse": True},
    "lineart_anime": {},
    "canny": {},
    "shuffle": {},
    "depth_zoe": {},
    "depth_leres": {"boost": False},
    "depth_leres++": {"boost": True},
    "mediapipe_face": {},
    "tile": {},
    "binary": {},
    "color": {},
}

CHOICES = f"Choices for the processor are {list(MODELS.keys())}"


class Processor:
    def __init__(self, processor_id: str, params: Optional[Dict] = None) -> None:
        """Processor that can be used to process images with controlnet aux processors

        Args:
            processor_id (str): processor name, options are 'hed, midas, mlsd, openpose,
                                pidinet, normalbae, lineart, lineart_coarse, lineart_anime,
                                canny, content_shuffle, zoe, mediapipe_face, tile'
            params (Optional[Dict]): parameters for the processor
        """
        LOGGER.info("Loading %s".format(processor_id))

        if processor_id not in MODELS:
            raise ValueError(
                f"{processor_id} is not a valid processor id. Please make sure to choose one of {', '.join(MODELS.keys())}"
            )

        self.processor_id = processor_id
        self.processor = self.load_processor(self.processor_id)

        # load default params
        self.params = MODEL_PARAMS[self.processor_id]
        # update with user params
        if params:
            self.params.update(params)

    def load_processor(self, processor_id: str) -> "Processor":
        """Load controlnet aux processors

        Args:
            processor_id (str): processor name

        Returns:
            Processor: controlnet aux processor
        """
        processor = MODELS[processor_id]["class"]

        if processor_id == "anime_face_segment":
            download_path = "bdsqlsz/qinglong_controlnet-lllite"
        elif "dw" in processor_id or processor_id == "animal_openpose":
            download_path = "yzd-v/DWPose"
        elif "densepose" in processor_id:
            download_path = "LayerNorm/DensePose-TorchScript-with-hint-image"
        elif "sam" in processor_id:
            download_path = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        else:
            download_path = "lllyasviel/Annotators"

        if MODELS[processor_id]["checkpoint"]:
            processor = processor.from_pretrained(download_path)
        elif "torchscript_device" in MODELS[processor_id]:
            processor = processor.from_pretrained(
                download_path, torchscript_device=DEVICE
            )
        elif "filename" in MODELS[processor_id]:
            processor = processor.from_pretrained(
                download_path, filename=MODELS[processor_id]["filename"]
            )
        else:
            processor = processor()

        if hasattr(processor, "to"):
            processor.to(DEVICE)

        return processor

    def __call__(
        self, image: Union[Image.Image, bytes], to_pil: bool = True
    ) -> Union[Image.Image, bytes]:
        """processes an image with a controlnet aux processor

        Args:
            image (Union[Image.Image, bytes]): input image in bytes or PIL Image
            to_pil (bool): whether to return bytes or PIL Image

        Returns:
            Union[Image.Image, bytes]: processed image in bytes or PIL Image
        """
        # check if bytes or PIL Image
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")

        if "depth_and_normal" in self.params:
            _, processed_image = self.processor(image, **self.params)
        else:
            processed_image = self.processor(image, **self.params)

        if to_pil:
            return processed_image
        else:
            output_bytes = io.BytesIO()
            processed_image.save(output_bytes, format="JPEG")
            return output_bytes.getvalue()
