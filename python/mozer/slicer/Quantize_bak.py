import tvm
import tvm.relay as relay
from tvm.relay.dataflow_pattern import *

# Profiling

class UnetPreProcessCallback(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        self.var2 = wildcard()
        tuple_node = is_tuple([wildcard(), self.var2])
        concat_node = is_op('concatenate')(tuple_node)
        self.pattern = concat_node
        self.match_node = []
        self.match_node2 = []

    def callback(self, pre, post, node_map):
        var2 = node_map[self.var2][0]
        self.match_node.append(var2)
        self.match_node2.append(pre)
        return pre 
        
class UnetCallback(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, match_node, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        self.tuple_get_item_node = is_tuple_get_item(wildcard(), 0)
        self.pattern_1 = self.tuple_get_item_node

        self.pattern = self.pattern_1 
        self.match_node = match_node
        self.counter = 0
        self.tmp = []

    def quant(self, node):
        # cast_to_int8 = relay.cast(
        #     relay.clip(
        #         relay.round(
        #             relay.multiply(node, relay.const(8.0))
        #         ), 
        #         a_min=-127.0, a_max=127.0
        #     ),
        #     dtype="int8"
        # )

        cast_to_int8 = relay.cast(
            relay.clip(
                relay.round(
                    relay.multiply(
                        relay.subtract(node, relay.const(18.0))
                        , relay.const(7.0))
                ), 
                a_min=-127.0, a_max=127.0
            ),
            dtype="int8"
        )
        result_node = relay.annotation.stop_fusion(cast_to_int8)
        self.tmp.append(result_node)
        return result_node

    def dequant(self, node):
        # cast_to_float32 = relay.divide(
        #     relay.cast(node, dtype='float32'), relay.const(8.0)
        # )
        cast_to_float32 = relay.add(
                relay.divide(
                relay.cast(node, dtype='float32'), relay.const(7.0))
            , relay.const(18.0)
        )
        return cast_to_float32

    def callback(self, pre, post, node_map):
        if self.pattern_1.match(pre):
            if pre in self.match_node:
                # print("pat 1")
                return self.dequant(self.quant(post))
        return post

class UnetCallback2(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, match_node, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        # self.tuple_get_item_node = is_tuple_get_item(wildcard(), 0)
        # self.pattern_1 = self.tuple_get_item_node
        self.var2 = wildcard()
        tuple_node = is_tuple([wildcard(), self.var2])
        concat_node = is_op('concatenate')(tuple_node)
        self.pattern = concat_node
        # self.pattern = self.pattern_1 
        self.match_node = match_node
        self.counter = 0
        self.tmp = []

    def quant(self, node):
        # cast_to_int8 = relay.cast(
        #     relay.clip(
        #         relay.round(
        #             relay.multiply(node, relay.const(8.0))
        #         ), 
        #         a_min=-127.0, a_max=127.0
        #     ),
        #     dtype="int8"
        # )
        cast_to_int8 = relay.cast(
            relay.clip(
                relay.round(
                    relay.multiply(
                        relay.subtract(node, relay.const(18.0))
                        , relay.const(7.0))
                ), 
                a_min=-127.0, a_max=127.0
            ),
            dtype="int8"
        )
        result_node = relay.annotation.stop_fusion(cast_to_int8)
        self.tmp.append(result_node)
        return result_node

    def dequant(self, node):
        # cast_to_float32 = relay.divide(
        #     relay.cast(node, dtype='float32'), relay.const(8.0)
        # )
        cast_to_float32 = relay.add(
                relay.divide(
                relay.cast(node, dtype='float32'), relay.const(7.0))
            , relay.const(18.0)
        )
        return cast_to_float32

    def callback(self, pre, post, node_map):
        if self.pattern.match(pre):
            if pre in self.match_node:
                # print("pat 1")
                return self.dequant(self.quant(post))
        return post


class UnetMaxPool2dCallback(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        max_pool2d_node = is_op('nn.max_pool2d')(wildcard())
        self.pattern = max_pool2d_node
        self.match_node = []

    def callback(self, pre, post, node_map):
        self.match_node.append(pre)
        return post


class UnetCallback3(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, match_node, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        # self.tuple_get_item_node = is_tuple_get_item(wildcard(), 0)
        # self.pattern_1 = self.tuple_get_item_node
        max_pool2d_node = is_op('nn.max_pool2d')(wildcard())
        self.pattern = max_pool2d_node
        self.match_node = match_node
        self.counter = 0
        self.tmp = []

    def quant(self, node):
        # cast_to_int8 = relay.cast(
        #     relay.clip(
        #         relay.round(
        #             relay.multiply(node, relay.const(8.0))
        #         ), 
        #         a_min=-127.0, a_max=127.0
        #     ),
        #     dtype="int8"
        # )
        cast_to_int8 = relay.cast(
            relay.clip(
                relay.round(
                    relay.multiply(
                        relay.subtract(node, relay.const(18.0))
                        , relay.const(7.0))
                ), 
                a_min=-127.0, a_max=127.0
            ),
            dtype="int8"
        )
        result_node = relay.annotation.stop_fusion(cast_to_int8)
        self.tmp.append(result_node)
        return result_node

    def dequant(self, node):
        # cast_to_float32 = relay.divide(
        #     relay.cast(node, dtype='float32'), relay.const(8.0)
        # )

        cast_to_float32 = relay.add(
                relay.divide(
                relay.cast(node, dtype='float32'), relay.const(7.0))
            , relay.const(18.0)
        )
        return cast_to_float32

    def callback(self, pre, post, node_map):
        # print("match pool2d")

        if self.pattern.match(pre):
            if pre in self.match_node:
                # print("pat 1")
                return self.dequant(self.quant(post))
        return post

class UnetLeakyReLUCallback(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        leaky_relu_node = is_op('nn.leaky_relu')(wildcard()) | is_op('nn.relu')(is_op('add')(wildcard(), wildcard()))
        self.pattern = leaky_relu_node
        self.match_node = []

    def callback(self, pre, post, node_map):
        self.match_node.append(pre)
        return post


class UnetCallback4(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, match_node, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        # self.tuple_get_item_node = is_tuple_get_item(wildcard(), 0)
        # self.pattern_1 = self.tuple_get_item_node
        leaky_relu_node = is_op('nn.leaky_relu')(wildcard())| is_op('nn.relu')(wildcard())
        self.pattern = leaky_relu_node
        self.match_node = match_node
        self.counter = 0
        self.tmp = []

    def quant(self, node):
        # cast_to_int8 = relay.cast(
        #     relay.clip(
        #         relay.round(
        #             relay.multiply(node, relay.const(8.0))
        #         ), 
        #         a_min=-127.0, a_max=127.0
        #     ),
        #     dtype="int8"
        # )

        cast_to_int8 = relay.cast(
            relay.clip(
                relay.round(
                    relay.multiply(
                        relay.subtract(node, relay.const(18.0))
                        , relay.const(7.0))
                ), 
                a_min=-127.0, a_max=127.0
            ),
            dtype="int8"
        )
        result_node = relay.annotation.stop_fusion(cast_to_int8)
        self.tmp.append(result_node)
        return result_node

    def dequant(self, node):
        # cast_to_float32 = relay.divide(
        #     relay.cast(node, dtype='float32'), relay.const(8.0)
        # )
        cast_to_float32 = relay.add(
                relay.divide(
                relay.cast(node, dtype='float32'), relay.const(7.0))
            , relay.const(18.0)
        )
        return cast_to_float32

    def callback(self, pre, post, node_map):
        # print("match leaky_relu_node")

        if self.pattern.match(pre):
            if pre in self.match_node:
                # print("pat 1")
                return self.dequant(self.quant(post))
        return post

# class AddCollector(DFPatternCallback):
#     # A callback class to rewrite the matched pattern to a batch_norm op.
#     def __init__(self, require_type=False):
#         super().__init__(require_type)
#         super().__init__(rewrite_once=True)

#         add_node = is_op('add')(wildcard(), wildcard())

#         self.pattern = add_node
#         self.match_node = []

#     def callback(self, pre, post, node_map):
#         # print(pre)
#         self.match_node.append(pre)
#         return post


class Int8Collector(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        int8_cast_node = is_op('cast')(wildcard()).has_attr({'dtype': 'int8'})

        self.pattern = int8_cast_node
        self.match_node = []

    def callback(self, pre, post, node_map):
        # print(pre)
        self.match_node.append(pre)
        return post


# TODO

# Profiling and resize the scale factor.

# get all int8 by Int8Collector

# The pattern would be

# def quant(self, node):
#     cast_to_int8 = relay.cast(
#         relay.clip(
#             relay.round(
#                 relay.multiply(node, relay.const(8.0))
#             ), 
#             a_min=-127.0, a_max=127.0
#         ),
#         dtype="int8"
#     )
#     result_node = relay.annotation.stop_fusion(cast_to_int8)
#     self.tmp.append(result_node)
#     return result_node

# def dequant(self, node):
#     cast_to_float32 = relay.divide(
#         relay.cast(node, dtype='float32'), relay.const(8.0)
#     )
#     return cast_to_float32

# def callback(self, pre, post, node_map):
#     if self.pattern.match(pre):
#         if pre in self.match_node:
#             # print("pat 1")
#             return self.dequant(self.quant(post))
#             # return self.dequant(
#             #         relay.layout_transform(
#             #         relay.layout_transform(
#             #             self.quant(post), src_layout='NCHW16c', dst_layout='NCHW'
#             #         ), src_layout='NCHW', dst_layout='NCHW16c')
#             #     )
#     return post

def quantize(mod, quantization_level):
    upc = UnetPreProcessCallback()
    out = rewrite(upc, mod['main'])
    if quantization_level == 0:
        maxpool = UnetMaxPool2dCallback()
        rewrite(maxpool, out)
        leakyrelu = UnetLeakyReLUCallback()
        rewrite(leakyrelu, out)
        callnodes = upc.match_node + upc.match_node2 + maxpool.match_node + leakyrelu.match_node + [out.body]
        callnodes_str = [str(node) for node in callnodes]
        callnodes_str = list(set(callnodes_str))
        callnodes_str.sort(key=lambda x: len(x))
        callnodes_str = callnodes_str[::-1]
        out_nodes = [None for i in range(len(callnodes_str))]
        for node in callnodes:
            out_nodes[callnodes_str.index(str(node))] = node
        # out = relay.Function(out.params, relay.Tuple(upc.match_node + upc.match_node2 + maxpool.match_node + [out.body]), out.ret_type, out.type_params, out.attrs)
        out = relay.Function(out.params, relay.Tuple(out_nodes), out.ret_type, out.type_params, out.attrs)
    else:
        uc = UnetCallback(upc.match_node)
        out = rewrite(uc, mod['main'])
        upc = UnetPreProcessCallback()
        rewrite(upc, out)
        uc2 = UnetCallback2(upc.match_node2)
        out = rewrite(uc2, out)
        
        if quantization_level == 1:
            callnodes = uc.tmp + [out.body]
            callnodes_str = [str(node) for node in callnodes]
            callnodes_str = list(set(callnodes_str))
            callnodes_str.sort(key=lambda x: len(x))
            callnodes_str = callnodes_str[::-1]
            out_nodes = [None for i in range(len(callnodes_str))]
            for node in callnodes:
                out_nodes[callnodes_str.index(str(node))] = node
            # out = relay.Function(out.params, relay.Tuple(uc.tmp + [out.body]), out.ret_type, out.type_params, out.attrs)
            out = relay.Function(out.params, relay.Tuple(out_nodes), out.ret_type, out.type_params, out.attrs)

        elif quantization_level == 2:

            upc = UnetMaxPool2dCallback()
            rewrite(upc, out)
            # print(len(upc.match_node))
            uc2 = UnetCallback3(upc.match_node)
            out = rewrite(uc2, out)

            upc = UnetLeakyReLUCallback()
            rewrite(upc, out)
            # print(len(upc.match_node))
            uc2 = UnetCallback4(upc.match_node)
            out = rewrite(uc2, out)
            int8_collector = Int8Collector()
            rewrite(int8_collector, out)
            
            callnodes = int8_collector.match_node + [out.body]
            callnodes_str = [str(node) for node in callnodes]
            callnodes_str = list(set(callnodes_str))
            callnodes_str.sort(key=lambda x: len(x))
            callnodes_str = callnodes_str[::-1]
            out_nodes = [None for i in range(len(callnodes_str))]
            for node in callnodes:
                out_nodes[callnodes_str.index(str(node))] = node
            # out = relay.Function(out.params, relay.Tuple(int8_collector.match_node + [out.body]), out.ret_type, out.type_params, out.attrs)
            out = relay.Function(out.params, relay.Tuple(out_nodes), out.ret_type, out.type_params, out.attrs)
    
    return out