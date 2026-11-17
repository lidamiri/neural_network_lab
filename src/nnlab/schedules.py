import math


class Schedule:
    name = "schedule"

    def __init__(self, base_lr):
        if float(base_lr) <= 0.0:
            raise ValueError("base learning rate must be positive")
        self.base_lr = float(base_lr)

    def value(self, epoch, total_epochs):
        raise NotImplementedError

    def get_config(self):
        return {"name": self.name, "base_lr": self.base_lr}


class Constant(Schedule):
    name = "constant"

    def value(self, epoch, total_epochs):
        return self.base_lr


class StepDecay(Schedule):
    name = "step"

    def __init__(self, base_lr, drop=0.5, every=10):
        super().__init__(base_lr)
        if not 0.0 < float(drop) <= 1.0:
            raise ValueError("drop factor must be in (0, 1]")
        if int(every) < 1:
            raise ValueError("every must be a positive integer")
        self.drop = float(drop)
        self.every = int(every)

    def value(self, epoch, total_epochs):
        return self.base_lr * self.drop ** (epoch // self.every)

    def get_config(self):
        config = super().get_config()
        config.update({"drop": self.drop, "every": self.every})
        return config


class ExponentialDecay(Schedule):
    name = "exponential"

    def __init__(self, base_lr, decay_rate=0.96):
        super().__init__(base_lr)
        if not 0.0 < float(decay_rate) <= 1.0:
            raise ValueError("decay_rate must be in (0, 1]")
        self.decay_rate = float(decay_rate)

    def value(self, epoch, total_epochs):
        return self.base_lr * self.decay_rate ** epoch

    def get_config(self):
        config = super().get_config()
        config.update({"decay_rate": self.decay_rate})
        return config


class CosineAnnealing(Schedule):
    name = "cosine"

    def __init__(self, base_lr, min_lr=0.0):
        super().__init__(base_lr)
        if float(min_lr) < 0.0 or float(min_lr) > self.base_lr:
            raise ValueError("min_lr must be between 0 and the base learning rate")
        self.min_lr = float(min_lr)

    def value(self, epoch, total_epochs):
        horizon = max(int(total_epochs) - 1, 1)
        progress = min(epoch / horizon, 1.0)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

    def get_config(self):
        config = super().get_config()
        config.update({"min_lr": self.min_lr})
        return config


_REGISTRY = {cls.name: cls for cls in (Constant, StepDecay, ExponentialDecay, CosineAnnealing)}


def available_schedules():
    return sorted(_REGISTRY)


def get_schedule(spec, base_lr):
    if isinstance(spec, Schedule):
        return spec
    if spec is None:
        return Constant(base_lr)
    if isinstance(spec, str):
        name, params = spec, {}
    elif isinstance(spec, dict):
        params = dict(spec)
        name = params.pop("name")
    else:
        raise TypeError(f"unsupported schedule specification: {spec!r}")
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown schedule '{name}', expected one of {available_schedules()}")
    return _REGISTRY[key](base_lr=base_lr, **params)
