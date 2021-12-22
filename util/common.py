class InferErr(Exception):
    def __init__(self, e):
        self.code = 1
        self.message = "Inference Error"
        super().__init__(self.message, str(e))


def MiB(val):
    return val * 1 << 20


def GiB(val):
    return val * 1 << 30


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()