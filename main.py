import pickle

with open("A:\MS_disffusion\data\crossdocked_pocket10\index.pkl","rb") as file:
    loaded_data = pickle.load(file)
    print(loaded_data)