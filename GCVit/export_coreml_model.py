# -*- coding: utf-8 -*-
import torch
import argparse
from timm.models import create_model
import coremltools as ct
from models import gc_vit

parser = argparse.ArgumentParser('Edge-ViT export CoreML model script', add_help=False)
parser.add_argument(
    '--batch-size',
    type=int,
    default=1,
    help='batch size used to export CoreML model.'
)
parser.add_argument(
    '--image-size',
    type=int,
    default=224,
    help='image size used to export CoreML model.'
)
parser.add_argument(
    '--model',
    type=str,
    help='model type'
)
args = parser.parse_args()

def main():
    model = create_model(
        args.model,
        num_classes=1000,
    )
    model.eval()
    input_tensor = torch.zeros((args.batch_size, 3, args.image_size, args.image_size), dtype=torch.float32)

    # Merge pre bn before exporting onnx/coreml model to speedup inference.
    if hasattr(model, "merge_bn"):
        model.merge_bn()

    coreml_file = "./%s_%dx%d.mlpackage" % (args.model, args.image_size, args.image_size)
    traced_model = torch.jit.trace(model, input_tensor)

    out = traced_model(input_tensor)
    model = ct.convert(
        traced_model, convert_to="mlprogram",
        inputs=[ct.ImageType(shape=input_tensor.shape)]
    )
    model.save(coreml_file)
    print("CoreML model saved to: %s."%coreml_file)
if __name__ == '__main__':
    main()
