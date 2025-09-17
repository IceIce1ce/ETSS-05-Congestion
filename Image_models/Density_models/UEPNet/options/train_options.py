from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # testing config
        parser.add_argument('--print_freq', type=int, default=100)
        parser.add_argument('--eval_freq', type=int, default=2)
        parser.add_argument('--save_latest_freq', type=int, default=5000)
        parser.add_argument('--continue_train', action='store_true')
        parser.add_argument('--epoch_count', type=int, default=1)
        parser.add_argument('--phase', type=str, default='train')
        # training config
        parser.add_argument('--n_epochs', type=int, default=2000)
        parser.add_argument('--n_epochs_decay', type=int, default=500)
        parser.add_argument('--beta1', default=0.9, type=float)
        parser.add_argument('--beta2', default=0.999, type=float)
        parser.add_argument('--lr', type=float, default=0.005)
        parser.add_argument('--lr_policy', type=str, default='step')
        parser.add_argument('--lr_decay_iters', type=int, default=5e10)
        self.isTrain = True
        return parser