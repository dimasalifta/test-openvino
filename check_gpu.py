from openvino.runtime import Core 

core = Core()
devices = core.available_devices
print("ada: ",devices)
