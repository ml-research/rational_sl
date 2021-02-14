import torchvision
import argparse


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--surgered_part', action="store", required=True,
                    type=str, help="Part to be surgered")

args = parser.parse_args()

resnet = torchvision.models.resnet101(pretrained=True).cuda()

# total_resnet_params = 0
# for elem in resnet.parameters():
#     total_resnet_params += elem.numel()
total_resnet_params = 44549160   # No need to be recomputed each time
print(f'{total_resnet_params:,}')

layer, block = args.surgered_part.split('.')
surgered = eval(f"resnet.layer{layer}[{block}]")

surgered_block_params = 0
for tensor in surgered.parameters():
    surgered_block_params += tensor.numel()

print(surgered)

print(f"Layer {layer} block {block} contains {surgered_block_params:,} parameters")
percent = surgered_block_params / total_resnet_params * 100
print(f"This correspond to {percent:.2f}%")
