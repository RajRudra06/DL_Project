import time
import data
import evaluate as E

train, val, test = data.load_splits()
print(len(train), len(val), len(test))