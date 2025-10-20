import time

import cloudpickle
import elements
import numpy as np
import portal


class Driver:
  mfunc = None
  def __init__(self, make_env_fns, parallel=True, **kwargs):
    assert len(make_env_fns) >= 1
    self.parallel = parallel
    self.kwargs = kwargs
    self.length = len(make_env_fns)
    if parallel:
      import multiprocessing as mp
      context = mp.get_context()
      self.pipes, pipes = zip(*[context.Pipe() for _ in range(self.length)])
      self.stop = context.Event()
      fns = [cloudpickle.dumps(fn) for fn in make_env_fns]
      self.procs = [
          portal.Process(self._env_server, self.stop, i, pipe, fn, start=True)
          for i, (fn, pipe) in enumerate(zip(fns, pipes))]
      self.pipes[0].send(('act_space',))
      self.act_space = self._receive(self.pipes[0])
      # MODIFICATION START: Fetch the action names from the remote environment.
      self.pipes[0].send(('get_attr', 'discrete_action_names'))
      self.action_names = self._receive(self.pipes[0]) or {}
      # MODIFICATION END
    else:
      self.envs = [fn() for fn in make_env_fns]
      self.act_space = self.envs[0].act_space
      # MODIFICATION START: Get action names from the local environment.
      self.action_names = getattr(self.envs[0], 'discrete_action_names', {})
      # MODIFICATION END
    self.callbacks = []
    self.acts = None
    self.carry = None
    self.reset()

  def set_callback(arg):
    Driver.mfunc = arg
    print(Driver.mfunc)
    return "ok"
  def reset(self, init_policy=None):
    self.acts = {
        k: np.zeros((self.length,) + v.shape, v.dtype)
        for k, v in self.act_space.items()}
    self.acts['reset'] = np.ones(self.length, bool)
    self.carry = init_policy and init_policy(self.length)

  def close(self):
    if self.parallel:
      [proc.kill() for proc in self.procs]
    else:
      [env.close() for env in self.envs]

  def on_step(self, callback):
    self.callbacks.append(callback)

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode)

  def _step(self, policy, step, episode):
    acts = self.acts
    acts = Driver.mfunc(acts);
    assert all(len(x) == self.length for x in acts.values())
    assert all(isinstance(v, np.ndarray) for v in acts.values())
    acts = [{k: v[i] for k, v in acts.items()} for i in range(self.length)]
    
    if self.parallel:
      [pipe.send(('step', act)) for pipe, act in zip(self.pipes, acts)]
      obs = [self._receive(pipe) for pipe in self.pipes]
    else:
      obs = [env.step(act) for env, act in zip(self.envs, acts)]
    obs = {k: np.stack([x[k] for x in obs]) for k in obs[0].keys()}
    logs = {k: v for k, v in obs.items() if k.startswith('log/')}
    obs = {k: v for k, v in obs.items() if not k.startswith('log/')}
    assert all(len(x) == self.length for x in obs.values()), obs
    self.carry, acts, outs = policy(self.carry, obs, **self.kwargs)
    
    # MODIFICATION START: detailed printing logic.
    prob_keys = [k for k in outs if k.endswith('_probs')]
    probs_list = []
    if prob_keys:
        # print("\n--- Model Action Probabilities ---")
        for key in prob_keys:
            action_name = key.replace('_probs', '')
            probabilities_np = np.array(outs[key])
            num_envs = probabilities_np.shape[0]
            
            # Get the list of names for this specific action, or use indices as a fallback.
            names = self.action_names.get(action_name, [str(i) for i in range(probabilities_np.shape[1])])

            # print(f"  Action '{action_name}':")
            for i in range(num_envs):
                # Pair names with probabilities and format them.
                probs_list = [(name, prob) for name, prob in zip(names, probabilities_np[i])]
                named_probs = [f"{name}:{prob:.2f}" for name, prob in zip(names, probabilities_np[i])]
                # print(f"    Env {i}: {{{', '.join(named_probs)}}}")
        # print("--------------------------------\n")
    # MODIFICATION END

    assert all(k not in acts for k in outs), (
        list(outs.keys()), list(acts.keys()))
    if obs['is_last'].any():
      mask = ~obs['is_last']
      acts = {k: self._mask(v, mask) for k, v in acts.items()}
    self.acts = {**acts, 'reset': obs['is_last'].copy()
    ,'_probs':probs_list,
    #prob_keys,
    '_mobs':obs
    }
    trans = {**obs, **acts, **outs, **logs}
    for i in range(self.length):
      trn = elements.tree.map(lambda x: x[i], trans)
      [fn(trn, i, **self.kwargs) for fn in self.callbacks]
    step += len(obs['is_first'])
    episode += obs['is_last'].sum()
    return step, episode

  def _mask(self, value, mask):
    while mask.ndim < value.ndim:
      mask = mask[..., None]
    return value * mask.astype(value.dtype)

  def _receive(self, pipe):
    try:
      msg, arg = pipe.recv()
      if msg == 'error':
        raise RuntimeError(arg)
      assert msg == 'result'
      return arg
    except Exception:
      print('Terminating workers due to an exception.')
      [proc.kill() for proc in self.procs]
      raise

  @staticmethod
  def _env_server(stop, envid, pipe, ctor):
    try:
      ctor = cloudpickle.loads(ctor)
      env = ctor()
      while not stop.is_set():
        if not pipe.poll(0.1):
          time.sleep(0.1)
          continue
        try:
          msg, *args = pipe.recv()
        except EOFError:
          return
        if msg == 'step':
          assert len(args) == 1
          act = args[0]
          obs = env.step(act)
          pipe.send(('result', obs))
        elif msg == 'obs_space':
          assert len(args) == 0
          pipe.send(('result', env.obs_space))
        elif msg == 'act_space':
          assert len(args) == 0
          pipe.send(('result', env.act_space))
        # MODIFICATION START: Add a handler to get arbitrary attributes from the env.
        elif msg == 'get_attr':
            attr_name = args[0]
            value = getattr(env, attr_name, None)
            pipe.send(('result', value))
        # MODIFICATION END
        else:
          raise ValueError(f'Invalid message {msg}')
    except ConnectionResetError:
      print('Connection to driver lost')
    except Exception as e:
      pipe.send(('error', e))
      raise
    finally:
      try:
        env.close()
      except Exception:
        pass
      pipe.close()
