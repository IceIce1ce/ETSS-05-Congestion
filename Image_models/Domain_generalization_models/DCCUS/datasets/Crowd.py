import os

def _pluck(root):
    img_path = root + '/img'
    ret = [filename for filename in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, filename))]
    return ret

class CrowdTest(object):
    def __init__(self, root):
        super(CrowdTest, self).__init__()
        self.root = root
        self.train = []
        self.load

    @property
    def load(self):
        self.train = _pluck(self.root)
        self.num_train = len(self.train)