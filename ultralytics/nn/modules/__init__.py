# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .block import (C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3,SimFusion_3in,SimFusion_4in,InjectionMultiSum_Auto_pool_defy,IFM,
                    PyramidPoolAgg,TopBasicLayer,AdvPoolFusion,C2f_DCNv2_Dynamic,C2f_DCNv2,SPPF_LSKA,
                    LAWDS,EMA,InjectionMultiSum_Auto_pool,RFB,C2f_FocusedLinearAttention,C2f_MSBlock,
                    CSPStage,Zoom_cat,Add,ScalSeq,C2f_MLCA,C2f_SCAM,C2f_SC,ScalSeq4,C2f_EMSCP,C2f_EMSC)
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention)
from .head import Classify, Detect, Pose, RTDETRDecoder, Segment, Detect_TADDH,DetectAux
# from .head import *
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)

__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect','Detect_TADDH',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP',
           'SimFusion_4in', 'SimFusion_3in', 'InjectionMultiSum_Auto_pool_defy', 'PyramidPoolAgg', 'TopBasicLayer', 'AdvPoolFusion'
           ,'C2f_DCNv2_Dynamic','C2f_DCNv2','SPPF_LSKA','LAWDS','EMA','InjectionMultiSum_Auto_pool','RFB',
           'C2f_FocusedLinearAttention','C2f_MSBlock','CSPStage','Zoom_cat','ScalSeq','Add',
           'C2f_MLCA','C2f_SCAM','C2f_SC','ScalSeq4','C2f_EMSCP','C2f_EMSC',)
