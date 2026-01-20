# from .assess import ASSESS
from .gtrans import GTrans
from .ema_teacher import EMATeacher
from .simple_tta import SimpleTTA, NoTTA
from .assess_node import ASSESSNode as ASSESS

tta_dict = {
    'assess': ASSESS,
    'gtrans': GTrans,
    'ema_teacher': EMATeacher,
    'simple_tta': SimpleTTA,
    'no_tta': NoTTA
}

def __all__():
    return ['ASSESS', 'GTrans', 'SimpleTTA', 'EMATeacher']
