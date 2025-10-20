import importlib
import os
import pathlib
import sys
from functools import partial as bind

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import elements
import embodied
import numpy as np
import portal
import ruamel.yaml as yaml




from importlib import reload
import embodied.core.driver as driver
import embodied.envs.crafter as crafter
reload(driver)


player_idx = 13

vitals = ["health","food","drink","energy",]

rot = np.array([[0,-1],[1,0]])
directions = ['front', 'right', 'back', 'left']



from crafter.env import Env

id_to_item = [0]*19
import itertools
dummyenv = Env()
for name, ind in itertools.chain(dummyenv._world._mat_ids.items(), dummyenv._sem_view._obj_ids.items()):
    name = str(name)[str(name).find('objects.')+len('objects.'):-2].lower() if 'objects.' in str(name) else str(name)
    id_to_item[ind] = name
player_idx = id_to_item.index('player')



def rotation_matrix(v1, v2):
    dot = np.dot(v1,v2)
    cross = np.cross(v1,v2)
    rotation_matrix = np.array([[dot, -cross],[cross, dot]])
    return rotation_matrix



def describe_loc(ref, P, noFacing=True):
    desc = []
    details = []
    if ref[1] > P[1]:
        desc.append("north")
    elif ref[1] < P[1]:
        desc.append("south")
    if ref[0] > P[0]:
        desc.append("west")
    elif ref[0] < P[0]:
        desc.append("east")

    if len(desc) == 2:
        if ref[1] > P[1]:
            details.append(f"{ref[1] - P[1]} step to the north")
        elif ref[1] < P[1]:
            details.append(f"{P[1] - ref[1]} step to the south")
        if ref[0] > P[0]:
            details.append(f"{ref[0] - P[0]} step to the west")
        elif ref[0] < P[0]:
            details.append(f"{P[0] - ref[0]} step to the east")
        pass
        
    elif len(desc) == 1:
        if ref[1] > P[1]:
            details.append(f"{ref[1] - P[1]} step to the north")
        elif ref[1] < P[1]:
            details.append(f"{P[1] - ref[1]} step to the south")
        if ref[0] > P[0]:
            details.append(f"{ref[0] - P[0]} step to the west")
        elif ref[0] < P[0]:
            details.append(f"{P[0] - ref[0]} step to the east")
        pass
    tmp = "-".join(desc)
    tmp2 = " and ".join(details)
    if noFacing == False:
        if "north" in tmp2:
            return "north"
        if "south" in tmp2:
            return "south"
        if "west" in tmp2:
            return "west"
        if "east" in tmp2:
            return "east"
        pass
    return tmp2


def describe_inventory(info):
    result = ""
    status_str = "Your status:\n{}".format("\n".join(["- {}: {}/9".format(v, info['inventory'][v]) for v in vitals]))
    result += status_str + "\n\n"
    inventory_str = "\n".join(["- {}: {}".format(i, num) for i,num in info['inventory'].items() if i not in vitals and num!=0])
    inventory_str = "Your inventory:\n{}".format(inventory_str) if inventory_str else "You have nothing in your inventory."
    result += inventory_str #+ "\n\n"
    return result.strip()



def rotation_matrix(v1, v2):
    dot = np.dot(v1,v2)
    cross = np.cross(v1,v2)
    rotation_matrix = np.array([[dot, -cross],[cross, dot]])
    return rotation_matrix

def describe_loc(ref, P, noFacing=True):
    desc = []
    details = []
    if ref[1] > P[1]:
        desc.append("north")
    elif ref[1] < P[1]:
        desc.append("south")
    if ref[0] > P[0]:
        desc.append("west")
    elif ref[0] < P[0]:
        desc.append("east")

    if len(desc) == 2:
        if ref[1] > P[1]:
            details.append(f"{ref[1] - P[1]} step to the north")
        elif ref[1] < P[1]:
            details.append(f"{P[1] - ref[1]} step to the south")
        if ref[0] > P[0]:
            details.append(f"{ref[0] - P[0]} step to the west")
        elif ref[0] < P[0]:
            details.append(f"{P[0] - ref[0]} step to the east")
        pass
        
    elif len(desc) == 1:
        if ref[1] > P[1]:
            details.append(f"{ref[1] - P[1]} step to the north")
        elif ref[1] < P[1]:
            details.append(f"{P[1] - ref[1]} step to the south")
        if ref[0] > P[0]:
            details.append(f"{ref[0] - P[0]} step to the west")
        elif ref[0] < P[0]:
            details.append(f"{P[0] - ref[0]} step to the east")
        pass

    
    tmp = "-".join(desc)
    tmp2 = " and ".join(details)

    if noFacing == False:
        if "north" in tmp2:
            return "north"
        if "south" in tmp2:
            return "south"
        if "west" in tmp2:
            return "west"
        if "east" in tmp2:
            return "east"
        pass
    return tmp2


def describe_env(info):
    assert(info['semantic'][info['player_pos'][0],info['player_pos'][1]] == player_idx)
    semantic = info['semantic'][info['player_pos'][0]-info['view'][0]//2:info['player_pos'][0]+info['view'][0]//2+1, info['player_pos'][1]-info['view'][1]//2+1:info['player_pos'][1]+info['view'][1]//2]
    center = np.array([info['view'][0]//2,info['view'][1]//2-1])
    result = ""
    x = np.arange(semantic.shape[1])
    y = np.arange(semantic.shape[0])
    x1, y1 = np.meshgrid(x,y)
    loc = np.stack((y1, x1),axis=-1)
    dist = np.absolute(center-loc).sum(axis=-1)
    obj_info_list = []
    
    facing = info['player_facing']
    target = (center[0] + facing[0], center[1] + facing[1])
    target = id_to_item[semantic[target]]
    obs = "You face {} at your front. your facing is: {}".format(target, describe_loc(np.array([0,0]),facing, False))
    

    #print(loc)

    for idx in np.unique(semantic):
        if idx==player_idx:
            continue

        smallest = np.unravel_index(np.argmin(np.where(semantic==idx, dist, np.inf)), semantic.shape)
        obj_info_list.append((id_to_item[idx], dist[smallest], describe_loc(np.array([0,0]), smallest-center) ) )

    if len(obj_info_list)>0:
        status_str = "You see:\n{}".format("\n".join(["- {} {}".format(name, loc) for name, dist, loc in obj_info_list]))
    else:
        status_str = "You see nothing away from you."
    result += status_str + "\n\n"
    result += obs.strip()
    
    return result.strip()


def describe_act(info):
    result = ""
    
    action_str = info['action'].replace('do_', 'interact_')
    action_str = action_str.replace('move_up', 'move_north')
    action_str = action_str.replace('move_down', 'move_south')
    action_str = action_str.replace('move_left', 'move_west')
    action_str = action_str.replace('move_right', 'move_east')
    
    act = "You took action {}.".format(action_str) 
    result+= act
    
    return result.strip()


def describe_status(info):
    if info['sleeping']:
        return "You are sleeping, and will not be able take actions until energy is full.\n\n"
    elif info['dead']:
        return "You died.\n\n"
    else:
        return ""

    
def describe_frame(info, action):
    try:
        result = ""
        
        if action is not None:
            result+=describe_act(info)
        result+=describe_status(info)
        result+="\n\n"
        result+=describe_env(info)
        result+="\n\n"
        result+=describe_inventory(info)
        
        return result.strip()
    except Exception as e:
        print(e)
        return "Error, you are out of the map."



def info_callback(args):
  print(args)
  description = describe_frame(args,None)
  print(description)
  print("wooow")
  #assert 1==3


def printTest(args):
  print(args)
  if '_probs' in args:
    del args['_probs']
  if '_mobs' in args:
    del args['_mobs']
  print("print test =>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  return args
print(
    driver.Driver.set_callback(printTest)
)
print(crafter.Crafter.set_info_func(info_callback))





def main(argv=None):
  from .agent import Agent
  [elements.print(line) for line in Agent.banner]

  configs = elements.Path(folder / 'configs.yaml').read()
  configs = yaml.YAML(typ='safe').load(configs)
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = elements.Flags(config).parse(other)
  config = config.update(logdir=(
      config.logdir.format(timestamp=elements.timestamp())))

  if 'JOB_COMPLETION_INDEX' in os.environ:
    config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))
  print('Replica:', config.replica, '/', config.replicas)

  logdir = elements.Path(config.logdir)
  print('Logdir:', logdir)
  print('Run script:', config.script)
  if not config.script.endswith(('_env', '_replay')):
    logdir.mkdir()
    config.save(logdir / 'config.yaml')

  def init():
    elements.timer.global_timer.enabled = config.logger.timer

  portal.setup(
      errfile=config.errfile and logdir / 'error',
      clientkw=dict(logging_color='cyan'),
      serverkw=dict(logging_color='cyan'),
      initfns=[init],
      ipv6=config.ipv6,
  )

  args = elements.Config(
      **config.run,
      replica=config.replica,
      replicas=config.replicas,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      report_length=config.report_length,
      consec_train=config.consec_train,
      consec_report=config.consec_report,
      replay_context=config.replay_context,
  )

  if config.script == 'train':
    embodied.run.train(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'train_eval':
    embodied.run.train_eval(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'eval_replay', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'eval_only':
    embodied.run.eval_only(
        bind(make_agent, config),
        bind(make_env, config),
        bind(make_logger, config),
        args)

  elif config.script == 'parallel':
    embodied.run.parallel.combined(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'replay_eval', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'parallel_env':
    is_eval = config.replica >= args.envs
    embodied.run.parallel.parallel_env(
        bind(make_env, config), config.replica, args, is_eval)

  elif config.script == 'parallel_envs':
    is_eval = config.replica >= args.envs
    embodied.run.parallel.parallel_envs(
        bind(make_env, config), bind(make_env, config), args)

  elif config.script == 'parallel_replay':
    embodied.run.parallel.parallel_replay(
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'replay_eval', 'eval'),
        bind(make_stream, config),
        args)

  else:
    raise NotImplementedError(config.script)


def make_agent(config):
  from .agent import Agent
  env = make_env(config, 0)
  notlog = lambda k: not k.startswith('log/')
  obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()
  if config.random_agent:
    return embodied.RandomAgent(obs_space, act_space)
  cpdir = elements.Path(config.logdir)
  cpdir = cpdir.parent if config.replicas > 1 else cpdir
  return Agent(obs_space, act_space, elements.Config(
      **config.agent,
      logdir=config.logdir,
      seed=config.seed,
      jax=config.jax,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      replay_context=config.replay_context,
      report_length=config.report_length,
      replica=config.replica,
      replicas=config.replicas,
  ))


def make_logger(config):
  step = elements.Counter()
  logdir = config.logdir
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  outputs = []
  outputs.append(elements.logger.TerminalOutput(config.logger.filter, 'Agent'))
  for output in config.logger.outputs:
    if output == 'jsonl':
      outputs.append(elements.logger.JSONLOutput(logdir, 'metrics.jsonl'))
      outputs.append(elements.logger.JSONLOutput(
          logdir, 'scores.jsonl', 'episode/score'))
    elif output == 'tensorboard':
      outputs.append(elements.logger.TensorBoardOutput(
          logdir, config.logger.fps))
    elif output == 'expa':
      exp = logdir.split('/')[-4]
      run = '/'.join(logdir.split('/')[-3:])
      proj = 'embodied' if logdir.startswith(('/cns/', 'gs://')) else 'debug'
      outputs.append(elements.logger.ExpaOutput(
          exp, run, proj, config.logger.user, config.flat))
    elif output == 'wandb':
      name = '/'.join(logdir.split('/')[-4:])
      outputs.append(elements.logger.WandBOutput(name))
    elif output == 'scope':
      outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))
    else:
      raise NotImplementedError(output)
  logger = elements.Logger(step, outputs, multiplier)
  return logger


def make_replay(config, folder, mode='train'):
  batlen = config.batch_length if mode == 'train' else config.report_length
  consec = config.consec_train if mode == 'train' else config.consec_report
  capacity = config.replay.size if mode == 'train' else config.replay.size / 10
  length = consec * batlen + config.replay_context
  assert config.batch_size * length <= capacity

  directory = elements.Path(config.logdir) / folder
  if config.replicas > 1:
    directory /= f'{config.replica:05}'
  kwargs = dict(
      length=length, capacity=int(capacity), online=config.replay.online,
      chunksize=config.replay.chunksize, directory=directory)

  if config.replay.fracs.uniform < 1 and mode == 'train':
    assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
        'Gradient scaling for low-precision training can produce invalid loss '
        'outputs that are incompatible with prioritized replay.')
    recency = 1.0 / np.arange(1, capacity + 1) ** config.replay.recexp
    selectors = embodied.replay.selectors
    kwargs['selector'] = selectors.Mixture(dict(
        uniform=selectors.Uniform(),
        priority=selectors.Prioritized(**config.replay.prio),
        recency=selectors.Recency(recency),
    ), config.replay.fracs)

  return embodied.replay.Replay(**kwargs)


def make_env(config, index, **overrides):
  suite, task = config.task.split('_', 1)
  if suite == 'memmaze':
    from embodied.envs import from_gym
    import memory_maze  # noqa
  ctor = {
      'dummy': 'embodied.envs.dummy:Dummy',
      'gym': 'embodied.envs.from_gym:FromGym',
      'dm': 'embodied.envs.from_dmenv:FromDM',
      'crafter': 'embodied.envs.crafter:Crafter',
      'dmc': 'embodied.envs.dmc:DMC',
      'atari': 'embodied.envs.atari:Atari',
      'atari100k': 'embodied.envs.atari:Atari',
      'dmlab': 'embodied.envs.dmlab:DMLab',
      'minecraft': 'embodied.envs.minecraft:Minecraft',
      'loconav': 'embodied.envs.loconav:LocoNav',
      'pinpad': 'embodied.envs.pinpad:PinPad',
      'langroom': 'embodied.envs.langroom:LangRoom',
      'procgen': 'embodied.envs.procgen:ProcGen',
      'bsuite': 'embodied.envs.bsuite:BSuite',
      'memmaze': lambda task, **kw: from_gym.FromGym(
          f'MemoryMaze-{task}-v0', **kw),
  }[suite]
  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)
  kwargs = config.env.get(suite, {})
  kwargs.update(overrides)
  if kwargs.pop('use_seed', False):
    kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
  if kwargs.pop('use_logdir', False):
    kwargs['logdir'] = elements.Path(config.logdir) / f'env{index}'
  env = ctor(task, **kwargs)
  return wrap_env(env, config)


def wrap_env(env, config):
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.NormalizeAction(env, name)
  env = embodied.wrappers.UnifyDtypes(env)
  env = embodied.wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.ClipAction(env, name)
  return env


def make_stream(config, replay, mode):
  fn = bind(replay.sample, config.batch_size, mode)
  stream = embodied.streams.Stateless(fn)
  stream = embodied.streams.Consec(
      stream,
      length=config.batch_length if mode == 'train' else config.report_length,
      consec=config.consec_train if mode == 'train' else config.consec_report,
      prefix=config.replay_context,
      strict=(mode == 'train'),
      contiguous=True)

  return stream


if __name__ == '__main__':
  main()
