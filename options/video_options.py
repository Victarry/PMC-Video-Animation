from .base_options import BaseOptions

class VideoOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        parser = self.parser
        parser.add_argument('--input_video', help='path of the input video')
        parser.add_argument('--phase', default='test-video')

    def parse(self):
        return self.parser.parse_args()