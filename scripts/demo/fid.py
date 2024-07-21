import pytorch_fid
import pytorch_fid.fid_score
import torch


def test_fid():

    paths = list()
    paths.append("E:\jikewct\Dataset\cifar10\cifar-10-images\\train_fid_stats.npz")
    paths.append("E:\jikewct\Repos\ml_code\data\\test\smld")
    dim = 2048
    fid_score = pytorch_fid.fid_score.calculate_fid_given_paths(
        paths,
        50,
        torch.device('cuda'),
        dim,
    )
    print(fid_score)


if __name__ == "__main__":
    test_fid()
