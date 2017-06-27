import mxnet as mx
import numpy
from mxnet.module.base_module import _parse_data_desc
from mxnet.module.module import Module
from mxnet.module.executor_group import DataParallelExecutorGroup
import mxnet.ndarray as nd
import sys


class DQNOutput(mx.operator.CustomOp):
    def __init__(modQ):
        super(DQNOutput, modQ).__init__()

    def forward(modQ, is_train, req, in_data, out_data, aux):
        modQ.assign(out_data[0], req[0], in_data[0])

    def backward(modQ, req, out_grad, in_data, out_data, in_grad, aux):
        out_qvalue = out_data[0].asnumpy()
        action = in_data[1].asnumpy().astype(numpy.int)
        target = in_data[2].asnumpy()
        ret = numpy.zeros(shape=out_qvalue.shape, dtype=numpy.float32)
        ret[numpy.arange(action.shape[0]), action] = numpy.clip(
            out_qvalue[numpy.arange(action.shape[0]), action] - target, -1, 1)
        modQ.assign(in_grad[0], req[0], ret)


@mx.operator.register("DQNOutput")
class DQNOutputProp(mx.operator.CustomOpProp):
    def __init__(modQ):
        super(DQNOutputProp, modQ).__init__(need_top_grad=False)

    def list_arguments(modQ):
        return ['data', 'action', 'target']

    def list_outputs(modQ):
        return ['output']

    def infer_shape(modQ, in_shape):
        data_shape = in_shape[0]
        action_shape = (in_shape[0][0],)
        target_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, action_shape, target_shape], [output_shape], []

    def create_operator(modQ, ctx, shapes, dtypes):
        return DQNOutput()


def sym(actions_num, signal_num, predict=False):
    signal = mx.symbol.Variable("signal")
    state = mx.symbol.Variable('data')
    net = mx.symbol.Concat(state, signal, name="Qnet_concat")
    data = mx.sym.Flatten(data=net)
    fc1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data=fc1, name='relu1', act_type="relu")
    fc2 = mx.symbol.FullyConnected(data=act1, name='fc2', num_hidden=64)
    act2 = mx.symbol.Activation(data=fc2, name='relu2', act_type="relu")
    Qvalue = mx.symbol.FullyConnected(data=act2, name='qvalue', num_hidden=actions_num)
    Dqn = mx.symbol.Custom(data=Qvalue, name='dqn', op_type='DQNOutput')
    com = mx.symbol.FullyConnected(data=act2, name='com', num_hidden=signal_num)
    signal = mx.symbol.Activation(data=com, name="signal", act_type="tanh")

    if predict:
        return mx.symbol.Group([signal, Qvalue ])
    else:
        return mx.symbol.Group([signal, Dqn])


import os


class Qnetwork():
    def __init__(self, actions_num, signal_num, dir, folder, q_ctx, bef_args=None, isTrain=True, batch_size=32):

        self.dir = dir
        self.folder = folder

        if not os.path.exists(dir + '/' + folder):
            os.makedirs(dir + '/' + folder)

        if isTrain:
            model = mx.mod.Module(symbol=sym(actions_num, signal_num),
                                  data_names=('data', 'signal', 'dqn_action', 'dqn_target'),
                                  label_names=None,
                                  context=q_ctx)
            bind(model,
                 data_shapes=[('data', (batch_size, 74)),
                              ('signal', (batch_size, signal_num)), ('dqn_action', (batch_size,)),
                              ('dqn_target', (batch_size,))], inputs_need_grad=True,
                 for_training=True)

            model.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34), arg_params=bef_args)
            model.init_optimizer(
                optimizer='adam',
                optimizer_params={
                    'learning_rate': 0.0002
                })
            self.model = model
        else:
            model = mx.mod.Module(symbol=sym(actions_num, signal_num, predict=True), data_names=('data', 'signal'),
                                  label_names=None,
                                  context=q_ctx)
            bind(model, data_shapes=[('data', (batch_size, 74)), ('signal', (batch_size, signal_num))],
                 for_training=False)
            model.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34), arg_params=bef_args)
            self.model = model

    def load_params(self, epoch):
        self.model.load_params(self.dir + '/' + self.folder + '/network-dqn_mx%04d.params' % epoch)

    def save_params(self, epoch):
        self.model.save_params(self.dir + '/' + self.folder + '/network-dqn_mx%04d.params' % epoch)


def copyTargetQNetwork(Qnet, target):
    arg_params, aux_params = Qnet.get_params()
    target.set_params(arg_params=arg_params, aux_params=aux_params)


def bind(modQ, data_shapes, label_shapes=None, for_training=True,
         inputs_need_grad=False, force_rebind=False, shared_module=None,
         grad_req='write'):
    if force_rebind:
        modQ._reset_bind()

    if modQ.binded:
        modQ.logger.warning('Already binded, ignoring bind()')
        return

    modQ.for_training = for_training
    modQ.inputs_need_grad = inputs_need_grad
    modQ.binded = True
    modQ._grad_req = grad_req

    if not for_training:
        assert not inputs_need_grad
    else:
        pass
        # this is not True, as some module might not contains a loss function
        # that consumes the labels
        # assert label_shapes is not None

    modQ._data_shapes, modQ._label_shapes = _parse_data_desc(
        modQ.data_names, modQ.label_names, data_shapes, label_shapes)

    if shared_module is not None:
        assert isinstance(shared_module, Module) and \
               shared_module.binded and shared_module.params_initialized
        shared_group = shared_module._exec_group
    else:
        shared_group = None

    modQ._exec_group = DataParallelExecutorGroup(modQ._symbol, modQ._context,
                                                 modQ._work_load_list, modQ._data_shapes,
                                                 modQ._label_shapes, modQ._param_names,
                                                 for_training, inputs_need_grad,
                                                 shared_group, logger=modQ.logger,
                                                 fixed_param_names=modQ._fixed_param_names,
                                                 grad_req=grad_req,
                                                 state_names=modQ._state_names)
    modQ._total_exec_bytes = modQ._exec_group._total_exec_bytes
    if shared_module is not None:
        modQ.params_initialized = True
        modQ._arg_params = shared_module._arg_params
        modQ._aux_params = shared_module._aux_params
    elif modQ.params_initialized:
        # if the parameters are already initialized, we are re-binding
        # so automatically copy the already initialized params
        modQ._exec_group.set_params(modQ._arg_params, modQ._aux_params)
    else:
        assert modQ._arg_params is None and modQ._aux_params is None
        param_arrays = [
            nd.zeros(x[0].shape, dtype=x[0].dtype, ctx=x[0][0].context)
            for x in modQ._exec_group.param_arrays
            ]
        modQ._arg_params = {name: arr for name, arr in zip(modQ._param_names, param_arrays)}

        aux_arrays = [
            nd.zeros(x[0].shape, dtype=x[0].dtype, ctx=x[0][0].context)
            for x in modQ._exec_group.aux_arrays
            ]
        modQ._aux_params = {name: arr for name, arr in zip(modQ._aux_names, aux_arrays)}

    if shared_module is not None and shared_module.optimizer_initialized:
        modQ.borrow_optimizer(shared_module)


class CDPG(object):
    def __init__(self, state_dim, signal_num, dir, folder, config):
        self.state_dim = state_dim
        self.num_envs = config.num_envs
        self.signal_num = signal_num
        self.config = config
        self.folder = folder
        self.dir = dir

        if not os.path.exists(dir + '/' + folder):
            os.makedirs(dir + '/' + folder)

        net = mx.sym.Variable('state')
        net = mx.sym.FullyConnected(
            data=net, name='fc1', num_hidden=100, no_bias=True)
        net = mx.sym.Activation(data=net, name='relu1', act_type="relu")
        signal = mx.symbol.FullyConnected(data=net, name='signal', num_hidden=signal_num)
        self.signal = mx.symbol.Activation(data=signal, name="com", act_type="tanh")
        self.model = mx.mod.Module(self.signal, data_names=('state',),
                                   label_names=None, context=config.ctx)

        self.paralell_num = config.num_envs * config.t_max
        bind(self.model, data_shapes=[('state', (self.paralell_num, state_dim))], label_shapes=None, grad_req="write")

        self.model.init_params(config.init_func)

        optimizer_params = {'learning_rate': config.learning_rate,
                            'rescale_grad': 1.0}
        if config.grad_clip:
            optimizer_params['clip_gradient'] = config.clip_magnitude

        self.model.init_optimizer(
            kvstore='local', optimizer=config.update_rule,
            optimizer_params=optimizer_params)

    def forward(self, state, is_train):
        self.model.reshape([('state', state.shape)])
        data_batch = mx.io.DataBatch(data=[mx.nd.array(state, ctx=self.config.ctx)], label=None)
        self.model.forward(data_batch, is_train=is_train)
        return self.model.get_outputs()

    def load_params(self, epoch):
        self.model.load_params(self.dir + '/' + self.folder + '/network-dqn_mx%04d.params' % epoch)

    def save_params(self, epoch):
        self.model.save_params(self.dir + '/' + self.folder + '/network-dqn_mx%04d.params' % epoch)
