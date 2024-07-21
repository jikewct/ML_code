import torch

def test_determinate():
    # | I + tA | = 1 + t* tr(A)
    I = torch.eye(10)
    A = torch.randn_like(I)
    for i in range(10):
        t = pow(1/10, i)
        determinate_value = torch.det(I + t * A)
        approximage_value = 1 + t * torch.trace(A)
        error_rate = ((determinate_value - approximage_value)/determinate_value).abs()
        print("t:{:.15f}, exact_value:{:.8f}, approx_value:{:.8f}, error_rate:{:.8f}".format(
            t, determinate_value, approximage_value, error_rate
        ))


if __name__ == "__main__":
    test_determinate()