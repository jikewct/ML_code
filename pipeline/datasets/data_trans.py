import torchvision.transforms as tforms


class ImageCenter:
    def __init__(self) -> None:
        pass

    def __call__(self, pic):
        #print(pic.shape, pic.max(),pic.min(), pic.mean())
        return 2 * pic - 1


trans = tforms.Compose([tforms.ToTensor(), ImageCenter()])

resize_trans = lambda img_size:  tforms.Compose([ tforms.Resize(img_size), tforms.ToTensor(), ImageCenter()])