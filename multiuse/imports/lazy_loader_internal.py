"""Lazy loader for internal imports."""


class LazyLoaderInternal:
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None

    def __getattr__(self, name):
        if self.module is None:
            self.module = __import__(self.module_name, fromlist=[name])
        return getattr(self.module, name)
