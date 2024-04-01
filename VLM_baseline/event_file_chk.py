import pickle as pkl
fn = 'data/coffee_events/step_1.pickle'
with open(fn, 'rb') as f:
    data = pkl.load(f)
print(type(data.metadata))
print(data.metadata.keys())