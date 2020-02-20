from experiments_nomissing.experiment_def import *
import pickle

dataset_name = 'car'
X = pickle.load(open('../data/car/X_car.pkl', 'rb'))
y = pickle.load(open('../data/car/y_car.pkl', 'rb'))

if not os.path.exists('results'):
    os.makedirs('results')

np.seterr(all='ignore')
for i in range(5, 100, 5):
    print(i)
    missingness = i/100
    with open('./results/{}_results_{}.txt'.format(dataset_name, i), 'w') as f:
        res = experiment_setting_1(X, y, missingness=missingness)
        f.writelines(str(res)[1:-1] + "\n")
        res = experiment_setting_2(X, y, missingness=missingness)
        f.writelines(str(res)[1:-1] + "\n")
        res = experiment_setting_3(X, y, missingness=missingness)
        f.writelines(str(res)[1:-1] + "\n")
        res = experiment_setting_4(X, y, missingness=missingness)
        f.writelines(str(res)[1:-1] + "\n")
        res = experiment_setting_5(X, y, missingness=missingness)
        f.writelines(str(res)[1:-1] + "\n")

