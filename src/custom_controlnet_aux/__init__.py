#Dummy file ensuring this package will be recognized
from .anime_face_segment import AnimeFaceSegmentor
from .binary import BinaryDetector
from .canny import CannyDetector
from .color import ColorDetector
from .densepose import DenseposeDetector
from .dwpose import DwposeDetector
from .dwpose import AnimalposeDetector
from .hed import HEDdetector
from .leres import LeresDetector
from .lineart import LineartDetector
from .lineart_anime import LineartAnimeDetector
from .manga_line import LineartMangaDetector
from .mediapipe_face import MediapipeFaceDetector
from .midas import MidasDetector
from .mlsd import MLSDdetector
from .normalbae import NormalBaeDetector
from .oneformer import OneformerSegmentor
from .open_pose import OpenposeDetector
from .pidi import PidiNetDetector
from .sam import SamDetector
from .scribble import ScribbleDetector, ScribbleXDog_Detector
from .shuffle import ColorShuffleDetector,ContentShuffleDetector,DownSampleDetector,GrayDetector,Image2MaskShuffleDetector
from .tile import TileDetector
from .uniformer import UniformerSegmentor
from .zoe import ZoeDetector
from .pyracanny import PyraCannyDetector
