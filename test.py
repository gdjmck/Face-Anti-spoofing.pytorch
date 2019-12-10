from train import Net
from torch.utils.data import DataLoader
from PIL import Image
import os
import torch
import torch.nn as nn
import argparse
import dataset
import dlib
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
    parser.add_argument('--ckpt', required=True, type=str, metavar='PATH',
                        help='checkpoint file for model')
    return parser.parse_args()

detector = dlib.get_frontal_face_detector()
def faceCrop(img):
    if type(img) == Image:
        img = np.array(img)
    try:
        face = detector(img)[0]
    except:
        return img
    height, width = img.shape[:2]
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    if w > width or h > height:
        return center_crop(img)
    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0
    if x+w > width:
        w = width - x
    if y+h > height:
        h = height - y
    if w > h:
        y -= (w-h)//2
        h = w
    else:
        x -= (h-w)//2
        w = h
    if (h <= 0) or (w <= 0) or (x < 0) or (y < 0):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        # 往外伸展不够，网内截取
        if w > h:
            x += (w-h)//2
            w = h
        else:
            y += (h-w)//2
            h = w
        if (h <= 0) or (w <= 0) or (x < 0) or (y < 0):
            return img
    return Image.fromarray(img[y: y+h, x: x+w, :])

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

def validate(device, net, val_loader, depth_dir = './depth_predict'):
    try:
        shutil.rmtree(depth_dir)
    except:
        pass
    try:
        os.makedirs(depth_dir)
    except:
        pass
    toImage = standard_transforms.ToPILImage(mode='L')
    net.eval()
    live_scores = []
    fake_scores = []
    for i, data in enumerate(val_loader):
        input, label = data
        img = invTrans(input)
        print(type(img), img.size())
        input = input.cuda(device)
        output, class_ret = net(input)
        out_depth = output[:,0,:,:]
        out_depth = out_depth.detach().cpu()
        class_ret = class_ret.detach().cpu()
        image = toImage(out_depth)
        class_output = nn.functional.softmax(class_ret, dim = 1)
        score = class_output[0][1]
        if label == 0:
            fake_scores.append(score)
            name = '' + depth_dir + '/fake-' + str(i) + '.bmp'
            image.save(name)
            img.save(name.replace('bmp', 'jpg'))

        if label == 1:
            live_scores.append(score)
            name = '' + depth_dir + '/live-' + str(i) + '.bmp'
            image.save(name)
            img.save(name.replace('bmp', 'jpg'))

    live_scores.sort()
    fake_scores.sort(reverse=True)
    fake_error = 0
    live_error = 0
    for val in fake_scores:
        if val >= 0.50:
            fake_error += 1
        else:
            break

    for val in live_scores:
        if val <= 0.50:
            live_error += 1
        else:
            break

    print('threshold 0.5: frp = ', fake_error / len(fake_scores),  '; tpr = ', (len(live_scores) - live_error) / len(live_scores))


if __name__ == '__main__':
    args = parse_args()
    test_live_rgb_dir = args.live_dir
    test_fake_rgb_dir = args.fake_dir
    depth_dir = args.depth_dir

    device = torch.device('cuda:0')
    net = Net()
    net = nn.DataParallel(net, device_ids = [0])
    net = net.to(device)
    
    assert os.path.isfile(args.ckpt)
    ckpt = torch.load(args.ckpt)
    net.load_state_dict(ckpt['state_dict'])

    normalize = standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    random_input_transform = standard_transforms.Compose([
        faceCrop,
        standard_transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1),
        standard_transforms.RandomResizedCrop((128, 128), scale=(0.9, 1), ratio=(1, 1)),
        standard_transforms.ToTensor(),
        normalize
    ])

    val_set = dataset.Dataset('test', test_live_rgb_dir, None, test_fake_rgb_dir,
        random_transform = random_input_transform, target_transform = None)
    val_loader = DataLoader(val_set, batch_size = 1, num_workers = 4, shuffle = False)

    validate(device, net, val_loader, depth_dir)