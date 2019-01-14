import nnvm
import tvm
import mxnet

import os
import argparse

parser = argparse.ArgumentParser(description='mxnet to tvm model convertor')
parser.add_argument('--target', default='broadwell', help='target cpu microarchitect')
parser.add_argument('--model', default='mneti', help='mxnet model prefix')
parser.add_argument('--shape', default='2688,1520', help='input model shape')
args = parser.parse_args()

batch_size = 1
input_shape = args.shape.split(',')
image_shape = (3, int(input_shape[1]), int(input_shape[0]))
data_shape = (batch_size,) + image_shape

prefix = './mxnet-model/' + args.model + '/' + args.model
epoch = 0

sym, arg_params, aux_params = mxnet.model.load_checkpoint(prefix, epoch)
nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(sym, arg_params, aux_params)

opt_level = 3
target = 'llvm -mcpu=' + args.target
target = tvm.target.create(target)
prefix = 'tvm-model/' + args.target
prefix +=  '/' + args.model + '/' + input_shape[0] + '_' + input_shape[1] + '/' 
os.system('mkdir -p ' + prefix)
#target_host = 'llvm -target=aarch64-linux-gnu'
#target_host = ''

with nnvm.compiler.build_config(opt_level=opt_level):
    graph, lib, params = nnvm.compiler.build(
        nnvm_sym, target=target, shape={"data": data_shape}, params=nnvm_params )

lib.export_library(prefix + "deploy_lib.so")
with open(prefix + "deploy_graph.json", "w") as fo:
    fo.write(graph.json())
with open(prefix + "deploy_param.params", "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))

