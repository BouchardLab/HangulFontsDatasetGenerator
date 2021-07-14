import numpy as np


def _random_params(rng, hp_space):
    params = {}
    for key, value in hp_space.items():
        size = value.get('size', None)
        tile = value.get('tile', None)
        if value['type'] == 'float':
            start = value['min']
            width = value['max']-start
            params[key] = width*rng.uniform(size=size)+start
        elif value['type'] == 'log10_float':
            start = value['min']
            width = value['max']-start
            params[key] = np.power(10., width*rng.uniform(size=size)+start)
        elif value['type'] == 'log10_1m_float':
            start = value['min']
            width = value['max']-start
            params[key] = 1. - np.power(10., width*rng.uniform(size=size)+start)
        elif value['type'] == 'int':
            low = value['min']
            high = value['max']
            params[key] = rng.randint(low=low, high=high+1, size=size)
        elif value['type'] == 'enum':
            params[key] = rng.choice(value['options'], size=size)
        elif value['type'] == 'list':
            low = value['min']
            high = value['max']
            if value['subtype'] == 'log10_float':
                num = np.power(10., rng.uniform(low=low, high=high))
                steps = np.geom(1e-4, num, num=26)
            elif value['subtype'] == 'log':
                num = np.log(rng.uniform(low=low, high=high))
                steps = np.arange(num, step=(num / 26)) # 26 is number of large epochs - number of loops of mini epochs, how long to stay at each C value for beta-vae

            else:
                raise ValueError("Bad type '"+str(value['type'])
                             +"' for parameter "+str(key)+'.')
            params[key] = steps
        else:
            raise ValueError("Bad type '"+str(value['type'])
                             +"' for parameter "+str(key)+'.')
        if tile is not None:
            params[key] = np.tile(params[key], tile)
    return params

class HyperparameterSpace(object):
    def __init__(self, input_shape, output_shape,
                 relu_only=True, include_resnets=False,
                 include_batch_norm=False,
                 info=False, seed=20170725):
        if isinstance(seed, int):
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = seed
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.relu_only = relu_only
        self.include_resnets = include_resnets
        self.include_batch_norm = include_batch_norm
        self.info = info
        self.input_dim = np.prod(self.input_shape)
        self.output_dim = np.prod(self.output_shape)

    def create_hp_space(self, hp_space=None):
        if hp_space is None:
            hp_space = {}

        # Model hyperparameters
        hp_space['dense_dim'] = {'type': 'int',
                                 'min': min(self.input_dim-1, self.output_dim * 10),
                                 'max': 2 * self.input_dim}
        hp_space['dense_dim_change'] = {'type': 'enum',
                                        'options': ['none']}
        if self.relu_only:
            hp_space['activation'] = {'type': 'enum',
                                      'options': ['relu']}
        else:
            hp_space['activation'] = {'type': 'enum',
                                      'options': ['relu', 'tanh', 'sigmoid']}
        if self.include_resnets:
            hp_space['use_resnet'] = {'type': 'int',
                                       'min': 1,
                                       'max': 1}
        if self.include_batch_norm:
            hp_space['use_batch_norm'] = {'type': 'int',
                                          'min': 0,
                                          'max': 1}
        if self.info:
            hp_space['info_beta'] = {'type': 'log10_float',
                                       'min': -4,
                                       'max': 0}

        # Optimization hyperparameters
        hp_space['lr'] = {'type': 'log10_float',
                          'min': -6,
                          'max': 1.}
        hp_space['momentum'] = {'type': 'log10_1m_float',
                                'min': -2,
                                'max': -0.004364805402450088} # momentum = .01
        hp_space['batch_size'] = {'type' : 'int',
                                  'min' : 32,
                                  'max' : 512, }
        hp_space['l2_wd'] = {'type': 'log10_float',
                                   'min': -6,
                                   'max': 1.}
        hp_space['input_dropout'] = {'type': 'float',
                                     'min': .01,
                                     'max': .9}
        hp_space['input_rescale'] = {'type': 'float',
                                     'min': .1,
                                     'max': 10.}
        hp_space['dense_dropout'] = {'type': 'float',
                                     'min': .01,
                                     'max': .9}
        hp_space['dense_rescale'] = {'type': 'float',
                                     'min': .1,
                                     'max': 10.}
        self.hp_space = hp_space
        return hp_space


    def random_params(self):
        return _random_params(self.rng, self.hp_space)

class DenseHyperparameterSpace(HyperparameterSpace):
    """Hyperparameter space for fully connected networks."""


    def __init__(self, input_shape, output_shape,
                 relu_only=True, include_resnets=False, info=False,
                 seed=20170725, allow_variable_dense_dim=False):
        super(DenseHyperparameterSpace, self).__init__(input_shape, output_shape,
                                                    relu_only=relu_only,
                                                    include_resnets=include_resnets,
                                                    info=info,
                                                    seed=seed)
        self.allow_variable_dense_dim = allow_variable_dense_dim


    def create_hp_space(self, hp_space=None):
        if hp_space is None:
            hp_space = {}
        hp_space = super(DenseHyperparameterSpace, self).create_hp_space(hp_space)

        # Model hyperparameters
        hp_space['n_dense_layers'] = {'type': 'int',
                                      'min': 2,
                                      'max': 7}
        if self.allow_variable_dense_dim:
            hp_space['dense_dim_change'] = {'type': 'enum',
                                            'options': ['none', 'linear', 'geometric']}
        else:
            hp_space['dense_dim_change'] = {'type': 'enum',
                                            'options': ['none']}
        return hp_space


class Conv2dHyperparameterSpace(HyperparameterSpace):
    """Hyperparameter space for 2d convnets."""


    def __init__(self, input_shape, output_shape,
                 relu_only=True, include_resnets=True, include_batch_norm=True,
                 seed=20170725, square_kernels=True):
        super(Conv2dHyperparameterSpace, self).__init__(input_shape, output_shape,
                                                    relu_only=relu_only,
                                                    include_resnets=include_resnets,
                                                    include_batch_norm=include_batch_norm,
                                                    seed=seed)
        self.square_kernels = square_kernels


    def create_hp_space(self, hp_space=None):
        if hp_space is None:
            hp_space = {}
        hp_space = super(Conv2dHyperparameterSpace, self).create_hp_space(hp_space)

        # Model hyperparameters
        if self.include_resnets :
            hp_space['n_conv_layers'] = {'type': 'int',
                                         'min': 4,
                                         'max': 15}
        else:
            hp_space['n_conv_layers'] = {'type': 'int',
                                         'min': 2,
                                         'max': 6}
        hp_space['input_kernel_size'] = {'type': 'int',
                                         'min': 2,
                                         'max': 7}
        hp_space['conv_kernel_size'] = {'type': 'int',
                                        'min': 2,
                                        'max': 5}
        hp_space['pool_size'] = {'type': 'int',
                                 'min': 2,
                                 'max': 4}
        if self.square_kernels:
            hp_space['input_kernel_size']['tile'] = 2
            hp_space['conv_kernel_size']['tile'] = 2
            hp_space['pool_size']['tile'] = 2
        else:
            hp_space['input_kernel_size']['size'] = 2
            hp_space['conv_kernel_size']['size'] = 2
            hp_space['pool_size']['size'] = 2
        hp_space['initial_kernel_number'] = {'type': 'int',
                                             'min': 8,
                                             'max': 64}
        hp_space['conv_dim_change'] = {'type': 'enum',
                                       'options': ['none', 'double',
                                                   'halve-first',
                                                   'halve-last']}
        hp_space['n_dense_layers'] = {'type': 'int',
                                      'min': 1,
                                      'max': 3}

        # Optimization hyperparameters
        hp_space['lr'] = {'type': 'log10_float',
                          'min': -5,
                          'max': -1}
        hp_space['batch_size'] = {'type' : 'int',
                                  'min' : 32,
                                  'max' : 256}
        hp_space['conv_dropout_type'] = {'type': 'enum',
                                         'options': ['conv', 'none', 'dropout']}
        hp_space['conv_dropout'] = {'type': 'float',
                                    'min': .01,
                                    'max': .9}
        hp_space['conv_rescale'] = {'type': 'float',
                                    'min': .1,
                                    'max': 10.}
        return hp_space

class VAEHyperparameterSpace(HyperparameterSpace):
    """Hyperparameter space for VAEs."""
    
    def __init__(self, input_shape, seed=11172020):
        super(VAEHyperparameterSpace, self).__init__(input_shape, input_shape,
                                                    relu_only=True,
                                                    include_resnets=True,
                                                    include_batch_norm=True,
                                                    seed=seed)
    def create_hp_space(self, hp_space=None):
        if hp_space is None:
            hp_space = {}
        hp_space = super(VAEHyperparameterSpace, self).create_hp_space(hp_space)
        
        hp_space['n_conv_layers'] = {'type': 'int',
                                     'min': 2,
                                     'max': 10}
        hp_space['input_kernel_size'] = {'type': 'int',
                                         'min': 3,
                                         'max': 15}
        hp_space['conv_kernel_size'] = {'type': 'int',
                                        'min': 2,
                                        'max': 8}
        hp_space['pool_size'] = {'type': 'int',
                                 'min': 2,
                                 'max': 4}
        hp_space['initial_kernel_number'] = {'type': 'int',
                                             'min': 8,
                                             'max': 64}
        hp_space['conv_dim_change'] = {'type': 'enum',
                                       'options': ['none', 'double',
                                                   'halve-first',
                                                   'halve-last']}
        hp_space['gamma'] = {'type': 'list',
                             'subtype': 'log10_float',
                             'min': 1,
                             'max': 3}
        hp_space['C'] = {'type': 'list',
                         'subtype': 'log',
                         'min': 20,
                         'max': 50}
        hp_space['n_dense_layers'] = {'type': 'int',
                                      'min': 1,
                                      'max': 3}
        hp_space['dense_dim'] = {'type': 'int',
                                 'min': 50,
                                 'max': 200}
        hp_space['h_dim'] = {'type': 'int',
                              'min': 20,
                              'max': 100}
        hp_space['batch_size'] = {'type': 'int',
                                 'min': 32,
                                 'max': 256}
        hp_space['input_kernel_size']['tile'] = 2
        hp_space['conv_kernel_size']['tile'] = 2
        hp_space['pool_size']['tile'] = 2
        
        hp_space['lr'] = {'type': 'log10_float',
                          'min': -5,
                          'max': -2}
        return hp_space   
    