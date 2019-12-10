from train import Net, validate
from torch.utils.data import DataLoader
import dataset
import torchvision.transforms as standard_transforms

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Liveness Testing')
    parser.add_argument('-s', '--scale', default=1.0, type=float,
                        metavar='N', help='net scale')
    parser.add_argument('--live_dir', required=True, type=str, metavar='PATH',
                        help='path to live image folder (required)')
    parser.add_argument('--fake_dir', required=True, type=str, metavar='PATH',
                        help='path to fake image folder (required)')
    parser.add_argument('--depth_dir', default='./depth', type=str, metavar='PATH',
                        help='path to save predicted depth image')
    return parser    

if __name__ == '__main__':
    args = parse_args()
    test_live_rgb_dir = args.live_dir
    test_fake_rgb_dir = args.fake_dir
    depth_dir = args.depth_dir

    device = torch.device('cuda:0')
    net = Net()
    net = nn.DataParallel(net, device_ids = [0])
    net = net.to(device)

    random_input_transform = standard_transforms.Compose([
        standard_transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1),
        standard_transforms.RandomResizedCrop((128, 128), scale=(0.9, 1), ratio=(1, 1)),
        standard_transforms.ToTensor(),
        normalize
    ])

    val_set = dataset.Dataset('test', test_live_rgb_dir, None, test_fake_rgb_dir,
        random_transform = random_input_transform, target_transform = target_transform)
    val_loader = DataLoader(val_set, batch_size = 1, num_workers = 4, shuffle = False)

    validate(device, net, val_loader, depth_dir)