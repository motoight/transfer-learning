import argparse
parser = argparse.ArgumentParser(description='Baby emotional detector')
parser.add_argument('--arch', type=str, choices=['VGG', 'Inception', 'ResNet50', 'Xception'])
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--resume', type=str, help='path to the resume model')
parser.add_argument('--data-root', type=str)
parser.add_argument('--out-dir', type=str)
parser.add_argument('--fix-encoder', action='store_true')
parser.add_argument('--use_dropout', action='store_true')
parser.add_argument('--use_labelsmooth', action='store_true')
parser.add_argument('--threshold', type=float, default=0.9)
parser.add_argument('--mu', type=int, default=7)
parser.add_argument('--lambda_u', type=int, default=5)
args = parser.parse_args()
writer_comment = "fixed={fix_encoder}_smooth={use_labelsmooth}_{bs}_{mu}".format(
    fix_encoder = args.fix_encoder,
        use_labelsmooth = args.use_labelsmooth,
            bs = args.batch_size,
                mu = args.mu)
print(writer_comment)

