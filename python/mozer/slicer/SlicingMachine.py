import json
import copy
from pyexpat import model
import numpy as np
from re import M
from sys import excepthook
from collections import defaultdict
from threading import currentThread
import tvm
from tvm.relay.testing import run_opt_pass
from tvm import relay
from tvm.relay import transform, build_module

# Graph Json Structure
#
# nodes
# arg_nodes
# heads
# attrs
# - dltype
# - device_index
# - storage_id
# - shape
# node_row_ptr


# RULE : return outputs by node number order
# return output node info and input node info (original node number)
# [ [graph1, [input node ("{original node},,,")], [output node] "{original node,,,}"], [graph2, [input node], [output node]],,, ]
# Sliced by range (start, end]
class TVMSlicer:
    #def __init__(self, graph_config='', slicing_point=[[]]):
    def __init__(self, graph_config=''):
        if isinstance(graph_config, str):
            try:
                graph_config = json.loads(graph_config)  
            except:
                return
        self.graph_config = copy.deepcopy(graph_config)

    def get_inputs(self):
        return [[i, g] for i, g in enumerate(zip(self.group, self.front_req, self.back_req))]

    def get_graph(self):
        return self.sliced_graph

    def get_mark(self):
        return self.dfs_list

    def slice_json_graph(self, start_nodes, end_nodes, is_quantize_sliced=False):
        graph_config = copy.deepcopy(self.graph_config)

        def dfs(cur_node_index, upper_bound, mark_list):
            # Already visited
            if cur_node_index in mark_list:
                return mark_list

            # Check upper bound
            if cur_node_index < 0:
                return mark_list
                
            if cur_node_index == upper_bound:
                mark_list.append(cur_node_index)
                return mark_list

            # Traverse
            mark_list.append(cur_node_index)
            input_lists = graph_config['nodes'][cur_node_index]['inputs']
            # print(input_lists)
            for input_node_index in input_lists:
                # print(cur_node_index, "->", input_node_index[0])
                mark_list = dfs(input_node_index[0], upper_bound, mark_list)
            return mark_list

        self.sliced_graph = []

        start_points = [start_node - 1 for start_node in start_nodes]
        # start_points.sort(reverse=True)
        end_points = [end_node for end_node in end_nodes]
        # start_points.sort(reverse=True)

        pre_nodes = []
        for start_point in start_points:
            nodes = np.array(sorted(dfs(start_point, 0, [])))
            pre_nodes = np.union1d(pre_nodes, nodes).astype(int)
            # pre_nodes = np.array(sorted(dfs(start_point, 0, [])))
        target_nodes = []
        for end_point in end_points:
            nodes = np.array(sorted(dfs(end_point, 0, [])))
            target_nodes = np.union1d(target_nodes, nodes).astype(int)
            # target_nodes = np.array(sorted(dfs(end_p, 0, [])))
        # print(target_nodes)
        # print("total_nodes")
        total_nodes = [i for i in range(len(graph_config['nodes']))]

        # model_nodes = target_nodes - pre_nodes 
        model_nodes = np.setdiff1d(target_nodes, pre_nodes)
        np.sort(model_nodes)
        
        # complement_nodes = total_nodes - model_nodes
        complement_nodes = pre_nodes
        # complement_nodes = np.setdiff1d(total_nodes, model_nodes)
        # complement_nodes = np.setdiff1d(total_nodes, target_nodes)

        # complement_nodes = total_nodes - model_nodes
        np.sort(complement_nodes)

        # ----------------------------------------
        # Check dependency of input nodes of model
        # ----------------------------------------

        intermediate_nodes = []
        # dep_input_info = defaultdict(list) # dep_node : model_nodes
        input_dependency = defaultdict(list) # input_node : model_node

        #####################
        # print("##############################")
        # print("Initial models")
        # print("pre_nodes", pre_nodes)
        # print("target_nodes", target_nodes)
        # print("model_nodes", model_nodes)
        # print("complement_nodes", complement_nodes)
        # print("##############################")

        for mnode in model_nodes:
            # Get all input nodes in 
            input_nodes = [e[0] for e in graph_config['nodes'][mnode]['inputs']]
            for inode in input_nodes:
                # if there is a input that is not included in model_nodes
                if inode not in model_nodes:
                    # if there is a input that is included in complement_nodes
                    if inode in complement_nodes:
                        input_input_nodes = [e[0] for e in graph_config['nodes'][inode]['inputs']]
                        ######## Quantized node check #########
                        if len(input_input_nodes) != 0 and graph_config["attrs"]["dltype"][1][inode] != 'int8':
                            for iinode in input_input_nodes:
                                iinode_dtype = graph_config["attrs"]["dltype"][1][iinode]
                                iinode_op = graph_config['nodes'][iinode]['op']
                                if iinode_dtype == 'int8' and iinode_op != 'null':
                                    intermediate_nodes.append(inode)
                                    input_dependency[iinode].append(inode)
                                else:
                                    input_dependency[inode].append(mnode)
                        else:
                            input_dependency[inode].append(mnode)

                    # Error : the dependency from nowhere!
                    else:
                        print("Unidentified dependency")
                

        # if len(intermediate_nodes) != 0:
        #     model_nodes = np.concatenate([model_nodes, np.array(intermediate_nodes)])
        #     np.sort(model_nodes)


        # ----------------------------------------
        # Check dependency of output nodes of model
        # ----------------------------------------

        # complement_nodes = post_nodes - model_nodes 
        complement_nodes = np.setdiff1d(total_nodes, target_nodes)
        complement_nodes = np.setdiff1d(complement_nodes, model_nodes)
        complement_nodes = np.setdiff1d(complement_nodes, intermediate_nodes)

        np.sort(complement_nodes)
        # print(complement_nodes)

        # print("######################################")
        # print("After input analyze")
        # print("intermediate_nodes", intermediate_nodes)
        # print("input_dependency", input_dependency)
        # print("model_nodes", model_nodes)
        # print("######################################")

        # Check dependency nodes of ex nodes - for output nodes in model nodes.
        # dep_output_info = defaultdict(list) # model_nodes : ex_dep_node
        output_dependency = defaultdict(list) # model_node : input_node
        for cnode in complement_nodes:
            input_nodes = [e[0] for e in graph_config['nodes'][cnode]['inputs']]
            for inode in input_nodes:
                if inode in model_nodes:
                    ######## Quantized node check #########
                    # Check the input of input. 
                    # If input of input is int8 -> We assume that that node has been quantized 
                    # We should export that node instead of orignally dependent node.
                    input_input_nodes = [e[0] for e in graph_config['nodes'][inode]['inputs']]
                    if len(input_input_nodes) != 0:
                        for iinode in input_input_nodes:
                            if iinode in model_nodes:
                                iinode_dtype = graph_config["attrs"]["dltype"][1][iinode]
                                iinode_op = graph_config['nodes'][iinode]['op']
                                if iinode_dtype == 'int8' and iinode_op != 'null' and graph_config["attrs"]["dltype"][1][inode] != 'int8':
                                    # print("######################")
                                    # print(input_input_nodes)
                                    # print("######################")
                                    # intermediate_nodes.append(inode)
                                    output_dependency[iinode].append(cnode)
                                else:
                                    output_dependency[inode].append(cnode)
                    else:
                        output_dependency[inode].append(cnode)

        # print("######################################")
        # print("After output analyze")
        # print("output_dependency", output_dependency)
        # print("model_nodes", model_nodes)
        # print("######################################")

        if len(intermediate_nodes) != 0:
            model_nodes = np.concatenate([model_nodes, np.array(intermediate_nodes)])
            np.sort(model_nodes)

        sliced_graph_config = {
            "nodes" : [],
            "arg_nodes": [],
            "heads": [],
            "attrs": { 
                "dltype": [
                    "list_str",
                    []
                ],
                "device_index": [
                    "list_int",
                    []
                ],
                "storage_id": [
                    "list_int",
                    []
                ],
                "shape": [
                    "list_shape",
                    []
                ],
            },
        "node_row_ptr": []
        }


        # print("input_dependency.keys()", input_dependency.keys())
        # Add input
        input_nodes = sorted(input_dependency.keys())

        for input_node_index in input_nodes:
            input_node_info = copy.deepcopy(graph_config['nodes'][input_node_index])
            input_node_info["op"] = "null"
            input_node_info["name"] = "input_{}".format(input_node_index)
            input_node_info["inputs"] = [] 
            sliced_graph_config["nodes"].append(input_node_info)
            sliced_graph_config["arg_nodes"].append(int(input_nodes.index(input_node_index)))
            sliced_graph_config["attrs"]["dltype"][1].append(graph_config["attrs"]["dltype"][1][input_node_index])
            sliced_graph_config["attrs"]["device_index"][1].append(graph_config["attrs"]["device_index"][1][input_node_index])
            sliced_graph_config["attrs"]["storage_id"][1].append(graph_config["attrs"]["storage_id"][1][input_node_index])
            sliced_graph_config["attrs"]["shape"][1].append(copy.deepcopy(graph_config["attrs"]["shape"][1][input_node_index]))
            sliced_graph_config["node_row_ptr"].append(int(input_node_index))


        # Add body
        model_nodes = sorted(model_nodes)
        model_nodes = input_nodes + model_nodes
        # print("model_nodes", model_nodes)

        for node_index in model_nodes[len(input_nodes):]:
            sliced_graph_config["nodes"].append(copy.deepcopy(graph_config['nodes'][node_index]))
            if graph_config["nodes"][node_index]["op"] == "null":
                sliced_graph_config["arg_nodes"].append(int(model_nodes.index(node_index)))
            sliced_graph_config["attrs"]["dltype"][1].append(graph_config["attrs"]["dltype"][1][node_index])
            sliced_graph_config["attrs"]["device_index"][1].append(graph_config["attrs"]["device_index"][1][node_index])
            sliced_graph_config["attrs"]["storage_id"][1].append(graph_config["attrs"]["storage_id"][1][node_index])
            sliced_graph_config["attrs"]["shape"][1].append(copy.deepcopy(graph_config["attrs"]["shape"][1][node_index]))
            sliced_graph_config["node_row_ptr"].append(int(node_index))

        # Set output
        output_nodes = sorted(output_dependency.keys())
        # When this chunck contain the tail of original model
        # TODO : Add logic for originally multiple output model.
        if len(output_nodes) == 0:
            output_nodes = [len(graph_config['nodes']) - 1]
        # print("output_dependency", output_dependency)
        # print("output_nodes", output_nodes)
        # Lookup Table for indexing.
        # {original_index : node_name}
        original_lut = dict()
        for idx, node_info in enumerate(graph_config['nodes']):
            name = node_info['name']
            original_lut[idx] = name

        # Lookup Table for indexing.
        # {node_name : new_index}
        lut = dict()
        for idx, node_info in enumerate(sliced_graph_config['nodes']):
            name = node_info['name']
            lut[name] = idx

        # print(lut)
        # Set input
        for node_index, input_index in enumerate(model_nodes):
            node_input_indexs = sliced_graph_config["nodes"][node_index]['inputs']
            for i, node in enumerate(node_input_indexs):
                try:
                    sliced_graph_config["nodes"][node_index]['inputs'][i] = [lut[original_lut[node[0]]], 0, 0]
                # when required node is transformed into input_{} 
                except:
                    # print('input_{}'.format(node[0]))
                    # print('name' , sliced_graph_config["nodes"][node_index]['name'])
                    sliced_graph_config["nodes"][node_index]['inputs'][i] = [lut['input_{}'.format(node[0])], 0, 0]
                    # except:
                    #     # parent int8
                    #     if 'input_{}'.format(node[0] - 1) in lut:
                    #         sliced_graph_config["nodes"][node_index]['inputs'][i] = [lut['input_{}'.format(node[0] - 1)], 0, 0]
                    #     else:
                    #         print("error")
                    #         print(lut)
                    #         continue
        output_nodes = np.setdiff1d(output_nodes, np.array([0]))
        for output_node_index in output_nodes:
            try:
                sliced_graph_config["heads"].append([lut[original_lut[output_node_index]], 0, 0])
            except:
                # print('input_{}'.format(node[0]))
                sliced_graph_config["heads"].append([lut['input_{}'.format(output_node_index)], 0, 0])


            # sliced_graph_config["heads"].append([output_node_index + len(input_nodes), 0, 0])

        # if len(sliced_graph_config["heads"]) == 0:
        #     sliced_graph_config["heads"].append([len(model_nodes) - 1, 0, 0])
        # Set rest : node_row_ptr
        sliced_graph_config["node_row_ptr"] = [i for i in range(len(sliced_graph_config["nodes"]) + 1)]
        # print("====================")

        # return [sliced_graph_config, input_nodes, [o + len(input_nodes) for o in output_nodes]]
        # print(output_dependency.keys())
        # print([h[0] for h in sliced_graph_config["heads"]])
        # print(output_nodes)
        return [sliced_graph_config, input_nodes, output_nodes.tolist()]


    def slice_relay_graph(self, expr, split_conf, params, is_quantize=False):
        """Splitting the graph into a list of subgraphs"""

        def dequant(node, scale=7.0, zero_point=18.0):
            deqnode = relay.cast(node, dtype='float32')
            deqnode = relay.divide(deqnode, relay.const(scale))
            deqnode = relay.add(deqnode, relay.const(zero_point))
            return deqnode

        def quant(node, scale=7.0, zero_point=18.0):
            qnode = relay.subtract(node, relay.const(zero_point))
            qnode = relay.multiply(qnode, relay.const(scale))
            qnode = relay.round(qnode)
            qnode = relay.clip(qnode, a_min=-128.0, a_max=127.0)
            qnode = relay.cast(qnode, dtype='int8')
            return qnode

        def get_dep_var(sub_var_dep):
            return [var for var in sub_var_dep[len(sub_var_dep) - 1]["ref_nodes"]]

        def parse_dependency(value, snode_dep, new_input_idx):
            new_args = []
            need_update = False
            for var in value.args:
                is_free_var = False
                for dep in snode_dep[:-1]:
                    if var in dep["nodes"]:
                        dep["nodes"][var] += 1
                        dep["ref_nodes"][var] = dep["nodes"][var]
                        is_free_var = True

                if is_free_var:
                    need_update = True
                    original_var = relay.var(f"{var.name_hint}", var.checked_type)
                    new_args.append(original_var)
                    new_input_idx += 1
                else:
                    new_args.append(var)

            if need_update:
                value = tvm.relay.expr.Call(
                    value.op, new_args, value.attrs, value.type_args, value.span
                )
            return value, snode_dep, new_input_idx

        def merge_constant_expr(constant_expr, expr):
            if not isinstance(constant_expr.body, tvm.relay.expr.Let):
                return tvm.relay.expr.Let(constant_expr.var, constant_expr.value, expr)

            return tvm.relay.expr.Let(
                constant_expr.var, constant_expr.value, merge_constant_expr(constant_expr.body, expr)
            )

        def _recursion(anf, pipeline_mods, split_conf, constant_expr):
            nonlocal operator_index_map
            nonlocal new_input_idx
            nonlocal snode_dep
            cur_node_dep = snode_dep[len(snode_dep) - 1]
            if isinstance(anf, tvm.relay.Function):
                return tvm.relay.Function(
                    anf.params,
                    _recursion(anf.body, pipeline_mods, split_conf, constant_expr),
                    anf.ret_type,
                    anf.type_params,
                    anf.attrs,
                )
            if isinstance(anf, tvm.relay.expr.Let):
                value = anf.value
                if isinstance(value, tvm.relay.expr.Constant):
                    if not constant_expr:
                        constant_expr = tvm.relay.expr.Let(anf.var, value, anf.var)
                    else:
                        constant_expr = tvm.relay.expr.Let(anf.var, value, constant_expr)
                if isinstance(value, tvm.relay.expr.Call):
                    new_args = []
                    # build current var list
                    cur_node_dep["nodes"][anf.var] = 0
                    # Get the dependency information of the nodes.
                    value, snode_dep, new_input_idx = parse_dependency(value, snode_dep, new_input_idx)
                    # need wraping dequant logic
                    # if need_quant:
                        
                    if isinstance(value.op, tvm.ir.Op):
                        if value.op.name in operator_index_map:
                            operator_index_map[value.op.name] += 1
                        else:
                            operator_index_map[value.op.name] = 0
                        split_operator_name = split_conf[0]["op_name"] if split_conf else ""
                        split_operator_index = split_conf[0]["op_index"] if split_conf else ""
                        # if a operator name and repeating count in the network match with the values
                        # of the 'split configuration', then this place is where we should do the
                        # graph splitting.
                        if (
                            split_conf
                            and split_operator_name in operator_index_map
                            and operator_index_map[split_operator_name] >= split_operator_index
                        ):
                            split_conf.pop(0)
                            snode_dep.append({"nodes": {}, "ref_nodes": {}})
                            ann = _recursion(
                                anf.body,
                                pipeline_mods,
                                split_conf,
                                constant_expr,
                            )
                            snode_dep.pop()
                            dep_vars = get_dep_var(snode_dep)
                            body = relay.Tuple(dep_vars) if len(dep_vars) > 1 else anf.var
                            if constant_expr:
                                ann = merge_constant_expr(constant_expr, ann)
                            pipeline_mods.insert(0, ann)
                            return tvm.relay.expr.Let(anf.var, value, body)
              
                return tvm.relay.expr.Let(
                    anf.var,
                    value,
                    _recursion(anf.body, pipeline_mods, split_conf, constant_expr),
                )
            else:
                return anf
        
        def getting_inputs(mod):
            return relay.analysis.free_vars(mod)

        def setting_outputs(anf, name_hints, outputs, names, is_quantize=False):
            if isinstance(anf, tvm.relay.Function):
                return tvm.relay.Function(
                    anf.params,
                    setting_outputs(anf.body, name_hints, outputs, names, is_quantize),
                    anf.ret_type,
                    anf.type_params,
                    anf.attrs,
                )
            if isinstance(anf, tvm.relay.expr.Let):
                value = anf.value
                if anf.var.name_hint in name_hints:
                    outputs.append(anf)
                return tvm.relay.expr.Let(
                    anf.var,
                    value,
                    setting_outputs(anf.body, name_hints, outputs, names, is_quantize),
                )
            else:
                new_outputs = []
                for o in outputs:
                    new_outputs.append(o.var)
                    names.append(o.var.name_hint)
                if isinstance(anf, tvm.relay.expr.Tuple):
                    for var in anf:
                        if var.name_hint not in names:
                            new_outputs.append(var)
                            names.append(var.name_hint)
                else:
                    if anf.name_hint not in names:
                        new_outputs.append(anf)
                        names.append(anf.name_hint)
                if is_quantize:
                    new_outputs = list(map(quant, new_outputs))
                new_map = tvm.relay.expr.Tuple(new_outputs)
                return new_map

        ################################################

        snode_dep = [{"nodes": {}, "ref_nodes": {}}]
        pipeline_mods = []
        operator_index_map = {}
        new_input_idx = 0
        constant_expr = None
        subgraph_split_conf = split_conf.copy()
        if params:
            expr = build_module.bind_params_by_name(expr, params)
        anf = run_opt_pass(expr, transform.ToANormalForm())
        anf = run_opt_pass(anf, transform.InferType())
        ann = _recursion(
            anf,
            pipeline_mods,
            subgraph_split_conf,
            constant_expr,
        )
        pipeline_mods.insert(0, ann.body)

        ################################################

        input_name_hints = []
        for idx, mod in enumerate(pipeline_mods):
            free_vars = getting_inputs(mod)
            new_input_vars = []
            if is_quantize and idx > 0:
                for free_var in free_vars:
                    new_input_var = tvm.relay.expr.Var(free_var.name_hint+"_deq", relay.TensorType(free_var.type_annotation.shape, 'int8'))
                    new_input_vars.append([free_var, dequant(new_input_var)])
                modmod = mod
                for ov, nv in new_input_vars:
                    new_anf = tvm.relay.expr.Let(ov, nv, modmod)
                    modmod = new_anf 
                pipeline_mods[idx] = modmod
            input_name_hints.append([free_var.name_hint for free_var in free_vars])
        ################################################

        total_input_name_hints = []
        for name_hints in input_name_hints:
            total_input_name_hints.extend(name_hints)
        
        output_name_hints = []
        for idx, mod in enumerate(pipeline_mods[:-1]):
            names = []
            out = setting_outputs(mod, total_input_name_hints, [], names, is_quantize)
            pipeline_mods[idx] = out
            output_name_hints.append(names)
        
        names = []
        out = setting_outputs(pipeline_mods[-1], total_input_name_hints, [], names, False)
        pipeline_mods[-1] = out
        output_name_hints.append(names)
            
        return pipeline_mods, input_name_hints, output_name_hints
        # return pipeline_mods, total_outputs

    def get_all_intermediate_node(self):
        intermediate_nodes = []
        for idx, node in enumerate(self.graph_config['nodes']):
            if len(node['inputs']) != 0:
                intermediate_nodes.append(idx)
        return intermediate_nodes

    def get_node_count(self, mod, pattern):
        """Count the number of occurrences of each operator in the module"""
        total_counter = 0
        def visit(node):
            nonlocal total_counter
            if pattern.match(node):
                total_counter += 1
        relay.analysis.post_order_visit(mod["main"], visit)
        return total_counter

    def slice_relay_graph_refactor(self, expr, split_conf, params, is_quantize=False):
        """Splitting the graph into a list of subgraphs"""

        def dequant(node, scale=7.0, zero_point=18.0):
            deqnode = relay.cast(node, dtype='float32')
            deqnode = relay.divide(deqnode, relay.const(scale))
            deqnode = relay.add(deqnode, relay.const(zero_point))
            return deqnode

        def quant(node, scale=7.0, zero_point=18.0):
            qnode = relay.subtract(node, relay.const(zero_point))
            qnode = relay.multiply(qnode, relay.const(scale))
            qnode = relay.round(qnode)
            qnode = relay.clip(qnode, a_min=-128.0, a_max=127.0)
            qnode = relay.cast(qnode, dtype='int8')
            return qnode

        def get_dep_var(sub_var_dep):
            return [var for var in sub_var_dep[len(sub_var_dep) - 1]["ref_nodes"]]

        def parse_dependency(value, snode_dep, new_input_idx):
            new_args = []
            need_update = False
            for var in value.args:
                is_free_var = False
                for dep in snode_dep[:-1]:
                    if var in dep["nodes"]:
                        dep["nodes"][var] += 1
                        dep["ref_nodes"][var] = dep["nodes"][var]
                        is_free_var = True

                if is_free_var:
                    need_update = True
                    original_var = relay.var(f"{var.name_hint}", var.checked_type)
                    new_args.append(original_var)
                    new_input_idx += 1
                else:
                    new_args.append(var)

            if need_update:
                value = tvm.relay.expr.Call(
                    value.op, new_args, value.attrs, value.type_args, value.span
                )
            return value, snode_dep, new_input_idx

        def merge_constant_expr(constant_expr, expr):
            if not isinstance(constant_expr.body, tvm.relay.expr.Let):
                return tvm.relay.expr.Let(constant_expr.var, constant_expr.value, expr)

            return tvm.relay.expr.Let(
                constant_expr.var, constant_expr.value, merge_constant_expr(constant_expr.body, expr)
            )

        def _recursion(anf, pipeline_mods, split_conf, constant_expr):
            nonlocal operator_index_map
            nonlocal new_input_idx
            nonlocal snode_dep
            cur_node_dep = snode_dep[len(snode_dep) - 1]
            if isinstance(anf, tvm.relay.Function):
                return tvm.relay.Function(
                    anf.params,
                    _recursion(anf.body, pipeline_mods, split_conf, constant_expr),
                    anf.ret_type,
                    anf.type_params,
                    anf.attrs,
                )
            if isinstance(anf, tvm.relay.expr.Let):
                value = anf.value
                if isinstance(value, tvm.relay.expr.Constant):
                    if not constant_expr:
                        constant_expr = tvm.relay.expr.Let(anf.var, value, anf.var)
                    else:
                        constant_expr = tvm.relay.expr.Let(anf.var, value, constant_expr)
                if isinstance(value, tvm.relay.expr.Call):
                    new_args = []
                    # build current var list
                    cur_node_dep["nodes"][anf.var] = 0
                    # Get the dependency information of the nodes.
                    value, snode_dep, new_input_idx = parse_dependency(value, snode_dep, new_input_idx)
                    # need wraping dequant logic
                    # if need_quant:
                        
                    if isinstance(value.op, tvm.ir.Op):
                        if value.op.name in operator_index_map:
                            operator_index_map[value.op.name] += 1
                        else:
                            operator_index_map[value.op.name] = 0
                        split_operator_name = split_conf[0]["op_name"] if split_conf else ""
                        split_operator_index = split_conf[0]["op_index"] if split_conf else ""
                        # if a operator name and repeating count in the network match with the values
                        # of the 'split configuration', then this place is where we should do the
                        # graph splitting.
                        if (
                            split_conf
                            and split_operator_name in operator_index_map
                            and operator_index_map[split_operator_name] >= split_operator_index
                        ):
                            split_conf.pop(0)
                            snode_dep.append({"nodes": {}, "ref_nodes": {}})
                            ann = _recursion(
                                anf.body,
                                pipeline_mods,
                                split_conf,
                                constant_expr,
                            )
                            snode_dep.pop()
                            dep_vars = get_dep_var(snode_dep)
                            body = relay.Tuple(dep_vars) if len(dep_vars) > 1 else anf.var
                            if constant_expr:
                                ann = merge_constant_expr(constant_expr, ann)
                            pipeline_mods.insert(0, ann)
                            return tvm.relay.expr.Let(anf.var, value, body)
              
                return tvm.relay.expr.Let(
                    anf.var,
                    value,
                    _recursion(anf.body, pipeline_mods, split_conf, constant_expr),
                )
            else:
                return anf
        
        def getting_inputs(mod):
            return relay.analysis.free_vars(mod)

        def setting_outputs(anf, name_hints, outputs, names, is_quantize=False):
            if isinstance(anf, tvm.relay.Function):
                return tvm.relay.Function(
                    anf.params,
                    setting_outputs(anf.body, name_hints, outputs, names, is_quantize),
                    anf.ret_type,
                    anf.type_params,
                    anf.attrs,
                )
            if isinstance(anf, tvm.relay.expr.Let):
                value = anf.value
                if anf.var.name_hint in name_hints:
                    outputs.append(anf)
                return tvm.relay.expr.Let(
                    anf.var,
                    value,
                    setting_outputs(anf.body, name_hints, outputs, names, is_quantize),
                )
            else:
                new_outputs = []
                for o in outputs:
                    new_outputs.append(o.var)
                    names.append(o.var.name_hint)
                if anf.name_hint not in names:
                    new_outputs.append(anf)
                    names.append(anf.name_hint)
                if is_quantize:
                    new_outputs = list(map(quant, new_outputs))
                new_map = tvm.relay.expr.Tuple(new_outputs)
                return new_map

        ################################################

        snode_dep = [{"nodes": {}, "ref_nodes": {}}]
        pipeline_mods = []
        operator_index_map = {}
        new_input_idx = 0
        constant_expr = None
        subgraph_split_conf = split_conf.copy()
        if params:
            expr = build_module.bind_params_by_name(expr, params)
        anf = run_opt_pass(expr, transform.ToANormalForm())
        anf = run_opt_pass(anf, transform.InferType())
        ann = _recursion(
            anf,
            pipeline_mods,
            subgraph_split_conf,
            constant_expr,
        )
        pipeline_mods.insert(0, ann.body)

        ################################################

        input_name_hints = []
        for idx, mod in enumerate(pipeline_mods):
            free_vars = getting_inputs(mod)
            new_input_vars = []
            if is_quantize and idx > 0:
                for free_var in free_vars:
                    new_input_var = tvm.relay.expr.Var(free_var.name_hint+"_deq", relay.TensorType(free_var.type_annotation.shape, 'int8'))
                    new_input_vars.append([free_var, dequant(new_input_var)])
                modmod = mod
                for ov, nv in new_input_vars:
                    new_anf = tvm.relay.expr.Let(ov, nv, modmod)
                    modmod = new_anf 
                pipeline_mods[idx] = modmod
            input_name_hints.append([free_var.name_hint for free_var in free_vars])

        ################################################

        total_input_name_hints = []
        for name_hints in input_name_hints:
            total_input_name_hints.extend(name_hints)
        
        output_name_hints = []
        for idx, mod in enumerate(pipeline_mods[:-1]):
            names = []
            out = setting_outputs(mod, total_input_name_hints, [], names, is_quantize)
            pipeline_mods[idx] = out
            output_name_hints.append(names)
        
        names = []
        out = setting_outputs(pipeline_mods[-1], total_input_name_hints, [], names, False)
        pipeline_mods[-1] = out
        output_name_hints.append(names)
            
        return pipeline_mods, input_name_hints, output_name_hints
        # return pipeline_mods, total_outputs