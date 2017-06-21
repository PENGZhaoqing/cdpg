import mxnet as mx
import numpy
import mxnet.ndarray as nd
from mxnet.context import Context
from mxnet.module.executor_group import DataParallelExecutorGroup
from mxnet.module.base_module import _parse_data_desc
from mxnet.module import Module
import os


class QNet():
    def __init__(self, state_dim, signal_num, act_space, dir, folder, config):

        self.state_dim = state_dim
        self.config = config
        self.act_space = act_space
        self.dir = dir
        self.folder = folder
        if not os.path.exists(dir + '/' + folder):
            os.makedirs(dir + '/' + folder)

        signal = mx.symbol.Variable("signal")
        state = mx.symbol.Variable('state')
        net = mx.symbol.Concat(state, signal, name="Qnet_concat")

        fc1 = mx.symbol.FullyConnected(data=net, name='fc1', num_hidden=128)
        act1 = mx.symbol.Activation(data=fc1, name='relu1', act_type="relu")
        fc2 = mx.symbol.FullyConnected(data=act1, name='fc2', num_hidden=64)
        act2 = mx.symbol.Activation(data=fc2, name='relu2', act_type="relu")
        qvalue = mx.symbol.FullyConnected(data=act2, name='qvalue', num_hidden=act_space)
        self.model = mx.mod.Module(qvalue, data_names=('state', 'signal'),
                                   label_names=None, context=self.config.ctx)
        self.paralell_num = config.num_envs * config.t_max
        bind(self.model,
             data_shapes=[('state', (self.paralell_num, state_dim)), ('signal', (self.paralell_num, signal_num))],
             label_shapes=None, inputs_need_grad=True,
             grad_req="write")

        self.model.init_params(config.init_func)

        optimizer_params = {'learning_rate': config.learning_rate,
                            'rescale_grad': 1.0}
        if config.grad_clip:
            optimizer_params['clip_gradient'] = config.clip_magnitude

        self.model.init_optimizer(
            kvstore='local', optimizer=config.update_rule,
            optimizer_params=optimizer_params)

    def forward(self, state, signal, is_train):
        self.model.reshape([('state', state.shape), ('signal', signal.shape)])
        data_batch = mx.io.DataBatch(
            data=[mx.nd.array(state, ctx=self.config.ctx), signal],
            label=None)
        self.model.forward(data_batch, is_train=is_train)
        return self.model.get_outputs()

    def load_params(self, epoch):
        self.model.load_params(self.dir + '/' + self.folder + '/network-dqn_mx%04d.params' % epoch)

    def save_params(self, epoch):
        self.model.save_params(self.dir + '/' + self.folder + '/network-dqn_mx%04d.params' % epoch)


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


def soft_copy_params_to(net, target, soft_target_tau):
    for name, arr in target.model._exec_group.execs[0].arg_dict.items():
        arr[:] = (1.0 - soft_target_tau) * arr[:] + \
                 soft_target_tau * net.model._exec_group.execs[0].arg_dict[name][:]


def copy_params_to(net, target):
    arg_params, aux_params = net.model.get_params()
    target.model.set_params(arg_params=arg_params, aux_params=aux_params)


def bind(module, data_shapes, label_shapes=None, for_training=True,
         inputs_need_grad=False, force_rebind=False, shared_module=None,
         grad_req='write'):
    # force rebinding is typically used when one want to switch from
    # training to prediction phase.
    if force_rebind:
        module._reset_bind()

    if module.binded:
        module.logger.warning('Already binded, ignoring bind()')
        return

    module.for_training = for_training
    module.inputs_need_grad = inputs_need_grad
    module.binded = True
    module._grad_req = grad_req

    if not for_training:
        assert not inputs_need_grad
    else:
        pass
        # this is not True, as some module might not contains a loss function
        # that consumes the labels
        # assert label_shapes is not None

    module._data_shapes, module._label_shapes = _parse_data_desc(
        module.data_names, module.label_names, data_shapes, label_shapes)

    if shared_module is not None:
        assert isinstance(shared_module, Module) and \
               shared_module.binded and shared_module.params_initialized
        shared_group = shared_module._exec_group
    else:
        shared_group = None

    module._exec_group = DataParallelExecutorGroup(module._symbol, module._context,
                                                   module._work_load_list, module._data_shapes,
                                                   module._label_shapes, module._param_names,
                                                   for_training, inputs_need_grad,
                                                   shared_group, logger=module.logger,
                                                   fixed_param_names=module._fixed_param_names,
                                                   grad_req=grad_req,
                                                   state_names=module._state_names)
    module._total_exec_bytes = module._exec_group._total_exec_bytes
    if shared_module is not None:
        module.params_initialized = True
        module._arg_params = shared_module._arg_params
        module._aux_params = shared_module._aux_params
    elif module.params_initialized:
        # if the parameters are already initialized, we are re-binding
        # so automatically copy the already initialized params
        module._exec_group.set_params(module._arg_params, module._aux_params)
    else:
        assert module._arg_params is None and module._aux_params is None
        param_arrays = [
            nd.zeros(x[0].shape, dtype=x[0].dtype, ctx=x[0][0].context)
            for x in module._exec_group.param_arrays
            ]
        module._arg_params = {name: arr for name, arr in zip(module._param_names, param_arrays)}

        aux_arrays = [
            nd.zeros(x[0].shape, dtype=x[0].dtype)
            for x in module._exec_group.aux_arrays
            ]
        module._aux_params = {name: arr for name, arr in zip(module._aux_names, aux_arrays)}

    if shared_module is not None and shared_module.optimizer_initialized:
        module.borrow_optimizer(shared_module)
