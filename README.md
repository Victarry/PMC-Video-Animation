# Unsupervised Coherent Video Cartoonization with Perceptual Motion Consistency
Unsupervised Coherent Video Cartoonization with Perceptual Motion Consistency. AAAI 2022. [Arxiv](https://arxiv.org/abs/2204.00795)

![Paper](assets/paper.png)

## Setup Environment
```bash
conda env create -f environment.yaml
conda activate video-animation
```

## Training
Coming soon.

## Testing
Download Pretrained Network from [google drive](https://drive.google.com/file/d/1nlqoQdlWIHz5aAHVQ-1TfKy3pnZ3Dthl).

### Translating Images
- Translate images in input directory and save into output directory.
```bash
python inference.py --input_path ${your_input_folder} --output_path ${your_output_folder} --model_path pretrained.ckpt
```

### Translating Video
```bash
python translate_video.py --input_video ${your_input_video} --output_dir ${your_output_folder} --model_path pretrained.ckpt
```

## Citation
```
@inproceedings{Liu2022UnsupervisedCV,
  title={Unsupervised Coherent Video Cartoonization with Perceptual Motion Consistency},
  author={Zhenhuan Liu and Liang Li and Huajie Jiang and Xin Jin and Dandan Tu and Shuhui Wang and Zhengjun Zha},
  booktitle={AAAI},
  year={2022}
}
```