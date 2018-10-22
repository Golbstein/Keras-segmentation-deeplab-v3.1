from keras.callbacks import *

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=50., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle', cycle_mult = 1):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.init_max_lr = max_lr
        self.step_size = step_size
        self.slope = (base_lr - max_lr)/step_size
        self.mode = mode
        self.gamma = gamma
        self.cycle_mult = cycle_mult
        self.cycle = 0
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(1.2**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'cosine':
                self.scale_fn = lambda x: np.cos(np.pi * x / step_size)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.iterations     = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        self.iterations = 1.
        self.step_size/=(2**self.cycle)
        self.cycle = 0
        if self.mode == 'cosine':
            self.scale_fn = lambda x: np.cos(np.pi * x / self.step_size)
        if self.mode == 'triangular2':
            self.max_lr = self.init_max_lr
            
    def clr(self):
        if (self.iterations % self.step_size)==1:
            self.iterations = 1.
            self.cycle+=1
            self.step_size*=self.cycle_mult
            if self.mode == 'triangular2':
                self.max_lr *= self.scale_fn(self.cycle)
            self.slope = (self.base_lr - self.max_lr)/self.step_size
            if self.mode == 'cosine':
                self.scale_fn = lambda x: np.cos(np.pi * x / self.step_size)
        
        if self.iterations == self.step_size:
            itr = self.iterations
        else:
            itr = self.iterations % self.step_size
        if self.scale_mode == 'cycle':
            return np.maximum(self.base_lr,
                              (self.slope * itr + self.max_lr))
        else:
            A = self.max_lr - self.base_lr
            return (A/2*(self.scale_fn(self.iterations) + 1) + self.base_lr) * self.gamma**(self.clr_iterations)
                
    def on_train_begin(self, logs={}):
        logs = logs or {}
        
        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.max_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        self.iterations     += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())
