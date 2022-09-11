from argparse import ArgumentParser
from models.model import WhiteBoxGANModel
from tqdm import tqdm
import torch
from utils.util import *
import cv2

def pre_transform(img):
    """Change numpy array of unit8 to torch tensors in range[-1, 1]"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # Normalize images range from (0, 1) to (-1, 1)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform(img)


def transfer_video(model, input_path, output_path):
    input_video = cv2.VideoCapture(input_path)
    frame_rate = input_video.get(cv2.CAP_PROP_FPS)
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    output_video = cv2.VideoWriter(
        str(output_path), fourcc, frame_rate, (width, height)
    )

    data = []
    for i in tqdm(range(1, length + 1)):
        ret, frame = input_video.read()  # BGR format
        ## transform
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = pre_transform(frame)
        data.append(img.cuda())
        if i % opt.batch_size == 0 or i == length:
            imgs = torch.stack(data)
            with torch.no_grad():
                with torch.autocast("cuda"):
                    styled_images = model(imgs)
            data = []
            ## inverse transform
            for j in range(imgs.shape[0]):
                styled_image = inverse_transform(styled_images[j])
                styled_frame = np.clip(
                    cv2.cvtColor(styled_image, cv2.COLOR_RGB2BGR), 0, 255
                )
                output_video.write(styled_frame)
    # Release everything if job is finished
    input_video.release()
    output_video.release()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", default="pretrained.ckpt")
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=32, type=int)
    opt = parser.parse_args()

    model = WhiteBoxGANModel.load_from_checkpoint(opt.model_path, strict=False)
    input_video = Path(opt.input_video)
    assert input_video.exists()

    output_path = Path(opt.output_dir) / input_video.name
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    generator = model.cuda()
    print(f"saving output video to {output_path}")
    transfer_video(generator, opt.input_video, output_path)
