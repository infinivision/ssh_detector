import nnvm
import tvm
import mxnet

import os
import argparse

parser = argparse.ArgumentParser(description='mxnet to tvm model convertor')

args = parser.parse_args()

#cpu_target_list = ['broadwell','haswell','ivybridge']
cpu_target_list = ['skylake']
model_list = ['mneti']
input_shape_list = [(2688,1520),(2592,1944),(960,960),(640,640),(480,480),(120,120)]

for cpu_target in cpu_target_list:
	for model in model_list:
		for input_shape in input_shape_list:
			batch_size = 1
			image_shape = (3, input_shape[1], input_shape[0])
			data_shape = (batch_size,) + image_shape
			prefix = './mxnet-model/' + model + '/' + model
			epoch = 0
			sym, arg_params, aux_params = mxnet.model.load_checkpoint(prefix, epoch)
			nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(sym, arg_params, aux_params)
			opt_level = 3
			target = 'llvm -mcpu=' + cpu_target
			target = tvm.target.create(target)
			prefix = 'tvm-model/' + cpu_target
			prefix +=  '/' + model + '/' + str(input_shape[0]) + '_' + str(input_shape[1]) + '/' 
			os.system('mkdir -p ' + prefix)
			with nnvm.compiler.build_config(opt_level=opt_level):
					graph, lib, params = nnvm.compiler.build(
							nnvm_sym, target=target, shape={"data": data_shape}, params=nnvm_params )
			lib.export_library(prefix + "deploy_lib.so")
			with open(prefix + "deploy_graph.json", "w") as fo:
					fo.write(graph.json())
			with open(prefix + "deploy_param.params", "wb") as fo:
					fo.write(nnvm.compiler.save_param_dict(params))

