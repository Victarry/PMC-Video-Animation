from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import torch
from data.base import ImageFolder
from torchvision import transforms
from models.whiteboxgan_model import WhiteBoxGANModel
import numpy as np

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--input_path", type=str, required=True)
    argparser.add_argument("--output_path", type=str, required=True)
    argparser.add_argument("--model_path", type=str, required=True)

    args = argparser.parse_args()


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    folder = ImageFolder(args.input_path, transform=transform, return_paths=True)
    model = WhiteBoxGANModel.load_from_checkpoint(args.model_path).cuda()

    for i in range(len(folder)):
        img, path = folder[i]
        
        torch.cuda.empty_cache()
        C, H, W = img.shape
        H = H // 4 * 4
        W = W // 4 * 4
        img = img[:, :H, :W].cuda()
        try:
            with torch.no_grad():
                output = model(img.unsqueeze(0))
            out_path = Path(args.output_path) / Path(path).name
            print(img.shape, path)
            output = torch.clamp(output[0].permute(1, 2, 0)*127.5+127.5, 0, 255)
            Image.fromarray(output.cpu().numpy().astype(np.uint8)).save(out_path)
        except Exception as e:
            print(e)
            print('Fail:', img.shape, path)
