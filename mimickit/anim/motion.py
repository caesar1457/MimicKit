import enum
import pickle

class LoopMode(enum.Enum):
    CLAMP = 0
    WRAP = 1

class CustomUnpickler(pickle.Unpickler):
    """自定义 Unpickler 来处理从 __main__ 模块序列化的 Motion 对象以及 numpy 版本兼容性"""
    def find_class(self, module, name):
        # 将 __main__.Motion 映射到当前模块的 Motion 类
        if module == '__main__' and name == 'Motion':
            return Motion
        # 将 __main__.LoopMode 映射到当前模块的 LoopMode 类
        elif module == '__main__' and name == 'LoopMode':
            return LoopMode
        # 处理 numpy 版本兼容性：numpy 2.x 使用 _core，numpy 1.x 使用 core
        elif module == 'numpy._core.multiarray':
            module = 'numpy.core.multiarray'
        elif module == 'numpy._core.numeric':
            module = 'numpy.core.numeric'
        elif module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        # 其他情况使用默认行为
        return super().find_class(module, name)

def load_motion(file):
    with open(file, "rb") as filestream:
        motion_data = CustomUnpickler(filestream).load()
    return motion_data

class Motion():
    def __init__(self, loop_mode, fps, frames):
        self.loop_mode = loop_mode
        self.fps = fps
        self.frames = frames
        return

    def save(self, out_file):
        with open(out_file, "wb") as out_f:
            pickle.dump(self, out_f)
        return

    def get_length(self):
        num_frames = self.frames.shape[0]
        motion_len = float(num_frames - 1) / self.fps
        return motion_len