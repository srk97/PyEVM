_HPARAMS = dict()


def register(fn):
  global _HPARAMS
  _HPARAMS[fn.__name__] = fn()
  return fn


def get_hparams(name):
  return _HPARAMS[name]