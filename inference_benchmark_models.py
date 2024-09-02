import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data_sets import FineGrainedBirdClassificationDataset, PartImageNetDataset
from load_model import load_model_pdisco
import argparse
from tqdm import tqdm
import copy
from utils.training_utils.engine_utils import load_state_dict_pdisco
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# fix all the randomness for reproducibility
torch.backends.cudnn.enabled = True
torch.manual_seed(0)
torch.cuda.manual_seed(0)

torch.set_float32_matmul_precision('high')


def parse_args():
    parser = argparse.ArgumentParser(description='Inference benchmark models')
    parser.add_argument('--model_arch', default='resnet50', type=str,
                        help='pick model architecture')
    parser.add_argument('--use_torchvision_resnet_model', default=False, action='store_true')

    # Data
    parser.add_argument('--data_path',
                        help='directory that contains cub files, must'
                             'contain folder "./images"', required=True)
    parser.add_argument('--image_sub_path', default='images', type=str, required=False)
    parser.add_argument('--dataset', default='cub', type=str)
    parser.add_argument('--anno_path_test', default='', type=str, required=False)
    # Model params
    parser.add_argument('--num_parts', help='number of parts to predict',
                        default=8, type=int)
    parser.add_argument('--image_size', default=448, type=int)
    parser.add_argument('--output_stride', default=32, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    # Modulation
    parser.add_argument('--modulation_type', default="original",
                        choices=["original", "layer_norm", "parallel_mlp", "parallel_mlp_no_bias",
                                 "parallel_mlp_no_act", "parallel_mlp_no_act_no_bias", "none"],
                        type=str)
    parser.add_argument('--modulation_orth', default=False, action='store_true',
                        help='use orthogonality loss on modulated features')
    # Part Dropout
    parser.add_argument('--part_dropout', default=0.0, type=float)

    # Add noise to vit output features
    parser.add_argument('--noise_variance', default=0.0, type=float)

    # Gumbel Softmax
    parser.add_argument('--gumbel_softmax', default=False, action='store_true')
    parser.add_argument('--gumbel_softmax_temperature', default=1.0, type=float)
    parser.add_argument('--gumbel_softmax_hard', default=False, action='store_true')

    # Model path
    parser.add_argument('--model_path', default=None, type=str)

    # Classifier type
    parser.add_argument('--classifier_type', default="linear",
                        choices=["linear", "independent_mlp"], type=str)

    args = parser.parse_args()
    return args


def benchmark(args):
    args.eval_only = True
    args.pretrained_start_weights = True
    height = args.image_size
    test_transforms = transforms.Compose([
        transforms.Resize(size=height),
        transforms.CenterCrop(size=height),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    # define dataset path
    if args.dataset == 'cub':
        cub_path = args.data_path
        # define dataset and loader
        eval_data = FineGrainedBirdClassificationDataset(cub_path, split=1, transform=test_transforms, mode='test')
        num_cls = eval_data.num_classes
    elif args.dataset == 'part_imagenet':
        # define dataset and loader
        eval_data = PartImageNetDataset(data_path=args.data_path, image_sub_path=args.image_sub_path,
                                        transform=test_transforms,
                                        annotation_file_path=args.anno_path_test,
                                        )
        num_cls = eval_data.num_classes
    elif args.dataset == 'flowers102':
        # define dataset and loader
        eval_data = datasets.Flowers102(root=args.data_path, split='test', transform=test_transforms)
        num_cls = len(set(eval_data._labels))
    else:
        raise ValueError('Dataset not supported.')
    # Load the model
    model = load_model_pdisco(args, num_cls)
    snapshot_data = torch.load(args.model_path, map_location=torch.device('cpu'), weights_only=True)
    if 'model_state' in snapshot_data:
        _, state_dict = load_state_dict_pdisco(snapshot_data)
    else:
        state_dict = copy.deepcopy(snapshot_data)
    model.load_state_dict(state_dict, strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)
    model = torch.compile(model, mode="reduce-overhead")
    test_loader = torch.utils.data.DataLoader(
        eval_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # Warmup
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc='Warmup'):
        images = batch[0].to(device)
        with torch.no_grad():
            output = model(images)
        if i == 100:
            break

    # Benchmark
    for idx in tqdm(range(100), desc="Inference benchmark"):
        with torch.no_grad():
            output = model(images)

    print("Inference benchmark done!")

    torch._dynamo.reset()


if __name__ == '__main__':
    args = parse_args()
    benchmark(args)
