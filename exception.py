class AlignError(Exception):
  def __init__(self, msg='can not align word and morpheme'):
    self.msg = msg

  def __str__(self):
    return self.msg


class UnknownModelTypeError(Exception):
  def __init__(self, msg='unsupported model'):
    self.msg = msg

  def __str__(self):
    return self.msg