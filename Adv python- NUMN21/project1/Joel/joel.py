from src import io_util

sys.path.append(".")
data = io_util.load_mnist("data", maximum=1000)
print(data)