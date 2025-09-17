from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--output_dir', type=str, default='saved_sha')
        parser.add_argument('--phase', type=str, default='test')
        parser.add_argument('--eval', action='store_true')
        self.isTrain = False
        return parser