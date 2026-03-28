from .nwpu_crowd import build as build_nwpu_crowd
from .st_crowd import build as build_st_counting

def build_dataset(image_set, args):
    if args.name == "nwpu_crowd":
        return build_nwpu_crowd(image_set, args)
    elif args.name == "st_crowd":
        return build_st_counting(image_set, args)
    else:
        print('This dataset does not exist')
        raise NotImplementedError