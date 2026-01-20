import pandas as pd
import sys

args = sys.argv[1:]
if len(args) == 0:
    sys.exit(-1)

d = pd.read_excel(f"./results/{args[0]}.xlsx")
print(d)
