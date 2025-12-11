import pyepo, inspect
import pyepo.func


print("\n=== pyepo.func 内容 ===")
print(dir(pyepo.func))

# 查看 SPOPlus 源码
from pyepo.func import SPOPlus
print("\n=== SPOPlus 源码 ===")
print(inspect.getsource(SPOPlus))