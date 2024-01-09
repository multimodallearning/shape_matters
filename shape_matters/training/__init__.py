from .config import *
#from .training import *
from .training_mlp_datasplit import *
from .training_points import train_point2image, train_mlp_point2fx, point_sampling, train_point2point
from .training_graphs import train_graph2image, train_mlp_graph2fx
from .training_image import train_image2image, train_mlp_image2fx, train_image2point
from .training_end2end import train_CNN_mlp_image2fx, train_DGCNN_mlp_graph2fx, train_Pointnet_mlp_point2fx
from .training_img2surf import train_imagevox2surf

