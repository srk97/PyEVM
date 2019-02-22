from .registry import register

class HParams():
  def __init__(self):
    self.amplification = 20
    self.levels = 3
    self.cutoff_low = 0.83
    self.cutoff_high = 1
    self.mode = None
    self.save_only_diff = False

@register
def evm_default():
  hps = HParams()
  hps.mode = ['magnify_motion','magnify_color']
  return hps

@register
def evm_10_04_3():
  hps = evm_default()
  hps.amplification = 10
  hps.cutoff_low = 0.4
  hps.cutoff_high = 3

  return hps

@register
def evm_150_233_267():
  hps = evm_default()
  hps.amplification = 150
  hps.cutoff_low = 2.33
  hps.cutoff_high = 2.67

  return hps

def evm_120_default():
  hps = evm_default()
  hps.amplification = 120

  return hps  